#!/usr/bin/env python
"""
Fast converter: AgiBotWorld → LeRobot v2.1 format.

Reads directly from tar archives (no pre-extraction needed).
Uses system `tar` for fast selective video extraction, bypassing Python's slow tarfile iteration.
Writes parquet + meta files directly. Supports checkpoint/resume.
"""

import argparse
import gc
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from agibot_utils.agibot_utils import get_task_info
from agibot_utils.config import AgiBotWorld_TASK_TYPE
from ray.runtime_env import RuntimeEnv

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_FPS = 30
PROGRESS_FILE = "meta/progress.jsonl"
CHECKPOINT_FILE = "meta/checkpoint.json"
SKIPPED_FILE = "meta/skipped.jsonl"
DEFAULT_MAX_TAR_READERS = 4


@ray.remote
class TarSemaphore:
    """Limits concurrent NFS tar reads across all Ray workers."""

    def __init__(self, max_concurrent):
        self._max = max_concurrent
        self._current = 0
        self._waiters = []

    async def acquire(self):
        import asyncio
        if self._current < self._max:
            self._current += 1
            return True
        fut = asyncio.get_event_loop().create_future()
        self._waiters.append(fut)
        await fut
        return True

    async def release(self):
        self._current -= 1
        if self._waiters and self._current < self._max:
            self._current += 1
            fut = self._waiters.pop(0)
            fut.set_result(True)


# ── checkpoint / resume ───────────────────────────────────────────────


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def _log_skipped(local_dir, task_stem, eid, reason):
    path = local_dir / SKIPPED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"task": task_stem, "episode_id": eid, "reason": reason}) + "\n")


def _save_progress_line(local_dir, line_dict):
    path = local_dir / PROGRESS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(line_dict, default=_json_default) + "\n")


def _save_checkpoint(local_dir, ep_count, global_frame_idx, task_to_idx, completed_eids):
    ckpt = {
        "completed_episodes": ep_count,
        "global_frame_idx": global_frame_idx,
        "task_to_idx": task_to_idx,
        "completed_eids": completed_eids,
    }
    path = local_dir / CHECKPOINT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    tmp.rename(path)


def _load_resume_state(local_dir):
    ckpt_path = local_dir / CHECKPOINT_FILE
    prog_path = local_dir / PROGRESS_FILE
    if not ckpt_path.exists():
        return None

    with open(ckpt_path) as f:
        ckpt = json.load(f)
    n = ckpt["completed_episodes"]
    if n == 0:
        return None

    all_progress = []
    if prog_path.exists():
        with open(prog_path) as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                try:
                    all_progress.append(json.loads(line))
                except json.JSONDecodeError:
                    break

    with open(prog_path, "w") as f:
        for entry in all_progress:
            f.write(json.dumps(entry, default=_json_default) + "\n")

    all_stats = []
    for entry in all_progress:
        stats = entry.get("stats", {})
        np_stats = {k: {sk: np.array(sv) for sk, sv in sv_dict.items()} for k, sv_dict in stats.items()}
        all_stats.append(np_stats)

    return {
        "all_progress": all_progress,
        "all_episode_stats": all_stats,
        "global_frame_idx": ckpt["global_frame_idx"],
        "task_to_idx": ckpt["task_to_idx"],
        "completed_eids": set(ckpt.get("completed_eids", [])),
    }


# ── proprio loading ──────────────────────────────────────────────────


def _build_proprio_index(proprio_tar_path):
    """Index proprio_stats.tar → {task_id_str: {eid_int: member_name}}."""
    index = {}
    with tarfile.open(proprio_tar_path, "r") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith("proprio_stats.h5"):
                continue
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            try:
                tid, eid = parts[0], int(parts[1])
            except ValueError:
                continue
            index.setdefault(tid, {})[eid] = member.name
    return index


def _parse_h5_state_action(f, config):
    """Parse state and action arrays from an open h5py File. Returns (state, action, num_frames) or None."""
    state = {}
    action = {}

    for key in config["states"]:
        state[f"observation.states.{key}"] = np.array(f["state/" + key.replace(".", "/")], dtype=np.float32)
    for key in config["actions"]:
        action[f"actions.{key}"] = np.array(f["action/" + key.replace(".", "/")], dtype=np.float32)

    num_frames = len(next(iter(state.values())))

    # Validate all state arrays have consistent length
    for key, arr in state.items():
        if len(arr) != num_frames and len(arr) != 0:
            return None
        if len(arr) == 0:
            config_key = key[len("observation.states."):]
            shape = tuple(config["states"][config_key]["shape"])
            state[key] = np.zeros((num_frames, *shape), dtype=np.float32)

    for action_key, action_value in action.items():
        if len(action_value) == 0:
            config_key = action_key[len("actions."):]
            shape = tuple(config["actions"][config_key]["shape"])
            action[action_key] = np.zeros((num_frames, *shape), dtype=np.float32)
        elif len(action_value) < num_frames:
            state_key = action_key.replace("actions", "state").replace(".", "/")
            new_action = np.array(f[state_key], dtype=np.float32).copy()
            idx_key = "/".join(
                list(action_key.replace("actions", "action").split(".")[:-1]) + ["index"]
            )
            action_idx = np.array(f[idx_key])
            if not action_idx.size:
                action_idx = np.array(f[idx_key.replace("end", "joint")])
            new_action[action_idx] = action_value
            action[action_key] = new_action
        elif len(action_value) > num_frames:
            return None

    return state, action, num_frames


def _load_h5_from_tar(proprio_tar_path, member_name, config):
    """Read h5 from tar into memory and parse."""
    with tarfile.open(proprio_tar_path, "r") as tf:
        fileobj = tf.extractfile(member_name)
        if fileobj is None:
            return None
        h5_bytes = fileobj.read()

    with h5py.File(io.BytesIO(h5_bytes), "r") as f:
        result = _parse_h5_state_action(f, config)
    if result is None:
        return None
    state, action, num_frames = result
    return {"num_frames": num_frames, "state": state, "action": action}


def _load_h5_from_file(proprio_dir, eid, config):
    """Load h5 from extracted file on disk."""
    h5_path = proprio_dir / str(eid) / "proprio_stats.h5"
    if not h5_path.exists():
        return None

    with h5py.File(h5_path, "r") as f:
        result = _parse_h5_state_action(f, config)
    if result is None:
        return None
    state, action, num_frames = result
    return {"num_frames": num_frames, "state": state, "action": action}


# ── fast video extraction from tar (system tar) ─────────────────────


def _extract_videos_from_tar(tar_path, staging_dir, image_keys):
    """
    Use system `tar` to extract only video mp4 files from an observation tar.
    Much faster than Python tarfile on NFS (C implementation, better I/O buffering).
    """
    staging_dir.mkdir(parents=True, exist_ok=True)

    patterns = []
    for cam in image_keys:
        if "sensor" in cam:
            patterns.append(f"*/tactile/{cam}.mp4")
        else:
            patterns.append(f"*/videos/{cam}_color.mp4")

    cmd = ["tar", "--wildcards", "-xf", str(tar_path), "-C", str(staging_dir)] + patterns
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    # exit code 2 = "not found" for some patterns (e.g., tactile patterns in gripper tar) — OK
    if result.returncode not in (0, 2):
        raise RuntimeError(f"tar extraction failed: {result.stderr[:500]}")


def _scan_staging_videos(staging_dir, image_keys):
    """Scan staging dir for extracted videos. Returns {eid: {vid_key: path}}."""
    episodes = {}
    for cam in image_keys:
        vid_key = f"observation.images.{cam}"
        if "sensor" in cam:
            glob_pattern = f"*/tactile/{cam}.mp4"
        else:
            glob_pattern = f"*/videos/{cam}_color.mp4"

        for path in staging_dir.glob(glob_pattern):
            parts = path.relative_to(staging_dir).parts
            try:
                eid = int(parts[0])
            except ValueError:
                continue
            episodes.setdefault(eid, {})[vid_key] = path

    return episodes


# ── video validation ──────────────────────────────────────────────────


def _check_video_valid(path):
    """Try to open and decode one frame to verify mp4 integrity."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return False
            for _ in container.decode(video=0):
                return True
        return False
    except Exception:
        return False


# ── stats helpers ─────────────────────────────────────────────────────


def _compute_array_stats(arr):
    return {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "count": np.array([len(arr)]),
    }


def _compute_video_stats(video_path, sample_n=10, spatial_stride=4):
    """Sample a few frames from video and compute per-channel stats."""
    try:
        frames = []
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            return None
        frames = np.stack(frames)
    except Exception:
        return None

    n = len(frames)
    indices = np.linspace(0, n - 1, min(sample_n, n)).astype(int)
    sampled = frames[indices, ::spatial_stride, ::spatial_stride, :]
    flat = sampled.reshape(-1, 3).astype(np.float32) / 255.0
    return {
        "min": flat.min(axis=0).reshape(3, 1, 1),
        "max": flat.max(axis=0).reshape(3, 1, 1),
        "mean": flat.mean(axis=0).reshape(3, 1, 1),
        "std": flat.std(axis=0).reshape(3, 1, 1),
        "count": np.array([len(sampled)]),
    }


def _stats_to_jsonable(stats):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()}


# ── parquet writer ────────────────────────────────────────────────────


def _write_episode_parquet(local_dir, ep_idx, ep_data):
    chunk = ep_idx // DEFAULT_CHUNK_SIZE
    arrays = {}
    for key, arr in ep_data.items():
        if arr.ndim == 1:
            if arr.dtype == np.float32:
                arrays[key] = pa.array(arr.astype(np.float64))
            elif arr.dtype in (np.int64, np.int32):
                arrays[key] = pa.array(arr.astype(np.int64))
            else:
                arrays[key] = pa.array(arr)
        else:
            flat_size = int(np.prod(arr.shape[1:]))
            arrays[key] = pa.FixedSizeListArray.from_arrays(
                pa.array(arr.reshape(-1).astype(np.float32)),
                list_size=flat_size,
            )
    table = pa.table(arrays)
    out_path = local_dir / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


# ── meta writers (v2.1) ──────────────────────────────────────────────


def _write_episodes_jsonl(local_dir, all_progress):
    out = local_dir / "meta/episodes.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for entry in all_progress:
            f.write(json.dumps({
                "episode_index": entry["episode_index"],
                "tasks": entry["tasks"],
                "length": entry["length"],
            }) + "\n")


def _write_episodes_stats_jsonl(local_dir, all_progress):
    out = local_dir / "meta/episodes_stats.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for entry in all_progress:
            f.write(json.dumps({
                "episode_index": entry["episode_index"],
                "stats": entry["stats"],
            }, default=_json_default) + "\n")


def _write_tasks_jsonl(local_dir, task_to_idx):
    sorted_tasks = sorted(task_to_idx.items(), key=lambda x: x[1])
    out = local_dir / "meta/tasks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for task, idx in sorted_tasks:
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")


def _write_info_json(local_dir, config, total_episodes, total_frames, total_tasks, fps):
    features = {}
    total_videos = 0

    for key, img_cfg in config["images"].items():
        if "depth" in key:
            continue
        feat_key = f"observation.images.{key}"
        shape = list(img_cfg["shape"])
        if img_cfg["dtype"] == "video":
            features[feat_key] = {
                "dtype": "video",
                "shape": shape,
                "names": img_cfg["names"],
                "info": {
                    "video.height": shape[0],
                    "video.width": shape[1],
                    "video.codec": "unknown",
                    "video.pix_fmt": "yuv420p",
                    "video.fps": fps,
                    "video.channels": shape[2],
                },
            }
            total_videos += total_episodes

    for key, state_cfg in config["states"].items():
        features[f"observation.states.{key}"] = {
            "dtype": state_cfg["dtype"],
            "shape": list(state_cfg["shape"]),
            "names": state_cfg["names"],
        }

    for key, action_cfg in config["actions"].items():
        features[f"actions.{key}"] = {
            "dtype": action_cfg["dtype"],
            "shape": list(action_cfg["shape"]),
            "names": action_cfg["names"],
        }

    for sys_key, dtype in [
        ("timestamp", "float32"), ("frame_index", "int64"),
        ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64"),
    ]:
        features[sys_key] = {"dtype": dtype, "shape": [1], "names": None}

    total_chunks = (total_episodes + DEFAULT_CHUNK_SIZE - 1) // DEFAULT_CHUNK_SIZE

    info = {
        "codebase_version": "v2.1",
        "robot_type": "a2d",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }

    out = local_dir / "meta/info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(info, f, indent=4)


def _write_stats_json(local_dir, all_episode_stats):
    if not all_episode_stats:
        return
    all_keys = list(all_episode_stats[0].keys())
    global_stats = {}

    for key in all_keys:
        ep_stats_list = [ep[key] for ep in all_episode_stats if key in ep]
        if not ep_stats_list:
            continue
        counts = np.array([s["count"][0] for s in ep_stats_list])
        total_count = counts.sum()

        means = np.stack([s["mean"] for s in ep_stats_list])
        variances = np.stack([s["std"] ** 2 for s in ep_stats_list])

        weights = counts.copy()
        while weights.ndim < means.ndim:
            weights = np.expand_dims(weights, axis=-1)

        total_mean = (means * weights).sum(axis=0) / total_count
        delta = means - total_mean
        total_var = ((variances + delta ** 2) * weights).sum(axis=0) / total_count

        agg = {
            "min": np.min(np.stack([s["min"] for s in ep_stats_list]), axis=0),
            "max": np.max(np.stack([s["max"] for s in ep_stats_list]), axis=0),
            "mean": total_mean,
            "std": np.sqrt(total_var),
            "count": np.array([int(total_count)]),
        }
        global_stats[key] = _stats_to_jsonable(agg)

    out = local_dir / "meta/stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(global_stats, f, indent=4)


# ── per-task conversion ──────────────────────────────────────────────


def get_all_tasks(src_path, output_path):
    for json_file in src_path.glob("task_info/*.json"):
        local_dir = output_path / "agibotworld" / json_file.stem
        yield (json_file, local_dir.resolve())


def save_as_lerobot_dataset_fast(config, task, skip_video_stats, staging_root=None, tar_sem=None):
    json_file, local_dir = task
    print(f"\n{'=' * 60}\nProcessing {json_file.stem} → {local_dir}\n{'=' * 60}", flush=True)

    src_path = json_file.parent.parent
    task_info = get_task_info(json_file)
    task_name = task_info[0]["task_name"]
    task_init_scene = task_info[0]["init_scene_text"]
    task_instruction = f"{task_name} | {task_init_scene}"
    task_id = json_file.stem.split("_")[-1]
    task_info_map = {ep["episode_id"]: ep for ep in task_info}

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    resume = _load_resume_state(local_dir)
    if resume:
        global_frame_idx = resume["global_frame_idx"]
        task_to_idx = resume["task_to_idx"]
        all_progress = resume["all_progress"]
        all_episode_stats = resume["all_episode_stats"]
        completed_eids = resume["completed_eids"]
        print(f"  Resuming: {len(all_progress)} episodes done, {global_frame_idx} frames", flush=True)
    else:
        global_frame_idx = 0
        task_to_idx = {}
        all_progress = []
        all_episode_stats = []
        completed_eids = set()

    if task_instruction not in task_to_idx:
        task_to_idx[task_instruction] = len(task_to_idx)
    task_idx_val = task_to_idx[task_instruction]

    image_keys = {k for k in config["images"] if "depth" not in k}
    expected_vid_keys = {f"observation.images.{k}" for k in image_keys}

    # Detect proprio source
    proprio_dir = src_path / f"proprio_stats/{task_id}"
    proprio_tar_path = src_path / "proprio_stats" / "proprio_stats.tar"
    proprio_from_tar = False
    proprio_index = {}

    if proprio_dir.is_dir() and any(proprio_dir.iterdir()):
        print(f"  Proprio: from extracted {proprio_dir}", flush=True)
    elif proprio_tar_path.is_file():
        proprio_from_tar = True
        print(f"  Proprio: from {proprio_tar_path.name} (indexing...)", flush=True)
        t_idx = time.time()
        full_index = _build_proprio_index(proprio_tar_path)
        proprio_index = full_index.get(task_id, {})
        print(f"  Proprio index: {len(proprio_index)} episodes ({time.time() - t_idx:.1f}s)", flush=True)
    else:
        print(f"  ERROR: no proprio data found for task {task_id}", flush=True)
        return

    # Find observation tars
    obs_tar_dir = src_path / f"observations/{task_id}"
    obs_tars = sorted(obs_tar_dir.glob("*.tar"))
    if not obs_tars:
        print(f"  No observation tars found in {obs_tar_dir}", flush=True)
        return

    if staging_root:
        staging_base = Path(staging_root) / f"agibot_staging_{os.getpid()}" / json_file.stem
    else:
        staging_base = local_dir / ".staging"
    staging_base.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    new_eps = 0

    for tar_path in obs_tars:
        tar_size_gb = tar_path.stat().st_size / (1024**3)
        t_tar = time.time()
        print(f"\n  --- Tar: {tar_path.name} ({tar_size_gb:.1f} GB) ---", flush=True)

        staging_dir = staging_base / tar_path.stem

        # Acquire semaphore to limit concurrent NFS reads
        if tar_sem is not None:
            ray.get(tar_sem.acquire.remote())

        try:
            _extract_videos_from_tar(tar_path, staging_dir, image_keys)
        except Exception as e:
            print(f"  ERROR extracting {tar_path.name}: {e}", flush=True)
            continue
        finally:
            if tar_sem is not None:
                ray.get(tar_sem.release.remote())

        t_extract = time.time()
        extract_speed = tar_size_gb / (t_extract - t_tar) if (t_extract - t_tar) > 0 else 0

        # Scan extracted videos
        staged_episodes = _scan_staging_videos(staging_dir, image_keys)
        print(
            f"  Extracted {len(staged_episodes)} episodes, {sum(len(v) for v in staged_episodes.values())} videos "
            f"({t_extract - t_tar:.1f}s, {extract_speed:.2f} GB/s)",
            flush=True,
        )

        for eid in sorted(staged_episodes):
            if eid in completed_eids:
                continue
            if eid not in task_info_map:
                continue

            vid_staging_paths = staged_episodes[eid]

            missing = expected_vid_keys - set(vid_staging_paths.keys())
            if missing:
                reason = f"missing_videos: {[k.split('.')[-1] for k in missing]}"
                print(f"  episode {eid}: {reason}, skipping", flush=True)
                _log_skipped(local_dir, json_file.stem, eid, reason)
                continue

            t_ep = time.time()
            ep_idx = len(all_progress)
            action_config = task_info_map[eid]["label_info"]["action_config"]

            # Load proprio
            if proprio_from_tar:
                if eid not in proprio_index:
                    print(f"  episode {eid}: no proprio in tar, skipping", flush=True)
                    _log_skipped(local_dir, json_file.stem, eid, "no_proprio_in_tar")
                    continue
                result = _load_h5_from_tar(proprio_tar_path, proprio_index[eid], config)
            else:
                result = _load_h5_from_file(proprio_dir, eid, config)

            if result is None:
                print(f"  episode {eid}: bad h5 data (inconsistent lengths), skipping", flush=True)
                _log_skipped(local_dir, json_file.stem, eid, "bad_h5_data")
                continue

            num_frames = result["num_frames"]
            state = result["state"]
            action = result["action"]

            # Validate extracted videos
            bad_vid = None
            for vid_key, vid_path in vid_staging_paths.items():
                if not _check_video_valid(vid_path):
                    bad_vid = vid_key
                    break
            if bad_vid:
                reason = f"corrupted_mp4: {bad_vid}"
                print(f"  episode {eid}: {reason}, skipping", flush=True)
                _log_skipped(local_dir, json_file.stem, eid, reason)
                continue

            # Move videos from staging to final output location
            ep_chunk = ep_idx // DEFAULT_CHUNK_SIZE
            vid_output_paths = {}
            for vid_key, staging_path in vid_staging_paths.items():
                dst = local_dir / f"videos/chunk-{ep_chunk:03d}/{vid_key}/episode_{ep_idx:06d}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staging_path), str(dst))
                vid_output_paths[vid_key] = dst
            t_vid_done = time.time()

            # Write parquet
            timestamps = np.arange(num_frames, dtype=np.float32) / DEFAULT_FPS
            frame_indices = np.arange(num_frames, dtype=np.int64)
            episode_indices = np.full(num_frames, ep_idx, dtype=np.int64)
            global_indices = np.arange(global_frame_idx, global_frame_idx + num_frames, dtype=np.int64)
            task_indices = np.full(num_frames, task_idx_val, dtype=np.int64)

            ep_data = {
                "timestamp": timestamps,
                "frame_index": frame_indices,
                "episode_index": episode_indices,
                "index": global_indices,
                "task_index": task_indices,
                **state,
                **action,
            }

            t_pq = time.time()
            _write_episode_parquet(local_dir, ep_idx, ep_data)
            t_pq_done = time.time()

            # Stats
            ep_stats = {}
            for key, arr in state.items():
                ep_stats[key] = _compute_array_stats(arr.reshape(num_frames, -1))
            for key, arr in action.items():
                ep_stats[key] = _compute_array_stats(arr.reshape(num_frames, -1))
            ep_stats["timestamp"] = _compute_array_stats(timestamps.reshape(-1, 1))
            ep_stats["frame_index"] = _compute_array_stats(frame_indices.astype(np.float64).reshape(-1, 1))
            ep_stats["episode_index"] = _compute_array_stats(episode_indices.astype(np.float64).reshape(-1, 1))
            ep_stats["index"] = _compute_array_stats(global_indices.astype(np.float64).reshape(-1, 1))
            ep_stats["task_index"] = _compute_array_stats(task_indices.astype(np.float64).reshape(-1, 1))

            if not skip_video_stats:
                for vid_key, vid_path in vid_output_paths.items():
                    vs = _compute_video_stats(vid_path)
                    if vs:
                        ep_stats[vid_key] = vs
            t_stats_done = time.time()

            all_episode_stats.append(ep_stats)

            progress_entry = {
                "episode_index": ep_idx,
                "tasks": [task_instruction],
                "length": num_frames,
                "stats": {k: _stats_to_jsonable(v) for k, v in ep_stats.items()},
                "action_config": action_config,
            }
            all_progress.append(progress_entry)
            global_frame_idx += num_frames
            completed_eids.add(eid)
            _save_progress_line(local_dir, progress_entry)

            new_eps += 1
            if new_eps % 10 == 0:
                _save_checkpoint(local_dir, len(all_progress), global_frame_idx, task_to_idx, sorted(completed_eids))

            elapsed = time.time() - t0
            eps_per_sec = new_eps / elapsed if elapsed > 0 else 0
            print(
                f"  ep {eid} (idx={ep_idx}) | {num_frames}f | "
                f"vid={t_vid_done - t_ep:.2f}s pq={t_pq_done - t_pq:.2f}s "
                f"stats={t_stats_done - t_pq_done:.2f}s | "
                f"{time.time() - t_ep:.1f}s | {eps_per_sec:.2f} ep/s",
                flush=True,
            )
            gc.collect()

        # Clean up staging for this tar
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    # Clean up staging base
    if staging_base.exists():
        shutil.rmtree(staging_base, ignore_errors=True)

    # Write final meta
    if all_progress:
        _write_episodes_jsonl(local_dir, all_progress)
        _write_episodes_stats_jsonl(local_dir, all_progress)
        _write_tasks_jsonl(local_dir, task_to_idx)
        _write_info_json(local_dir, config, len(all_progress), global_frame_idx, len(task_to_idx), DEFAULT_FPS)
        _write_stats_json(local_dir, all_episode_stats)
        _save_checkpoint(local_dir, len(all_progress), global_frame_idx, task_to_idx, sorted(completed_eids))

    skipped_path = local_dir / SKIPPED_FILE
    n_skipped = 0
    if skipped_path.exists():
        with open(skipped_path) as f:
            n_skipped = sum(1 for _ in f)
    print(
        f"\nDone {json_file.stem}: {len(all_progress)} episodes, {global_frame_idx} frames, "
        f"{n_skipped} skipped (see {SKIPPED_FILE})",
        flush=True,
    )


# ── main ──────────────────────────────────────────────────────────────


def main(
    src_path: str,
    output_path: str,
    eef_type: str,
    task_ids: list,
    cpus_per_task: int,
    skip_video_stats: bool,
    staging_dir: str = None,
    max_tar_readers: int = DEFAULT_MAX_TAR_READERS,
    debug: bool = False,
):
    src_path = Path(src_path)
    output_path = Path(output_path)
    tasks = list(get_all_tasks(src_path, output_path))

    staging_root = staging_dir or tempfile.gettempdir()
    print(f"Staging root: {staging_root} (local disk)", flush=True)

    config, type_task_ids = (
        AgiBotWorld_TASK_TYPE[eef_type]["task_config"],
        AgiBotWorld_TASK_TYPE[eef_type]["task_ids"],
    )

    if eef_type == "gripper":
        remaining_ids = AgiBotWorld_TASK_TYPE["dexhand"]["task_ids"] + AgiBotWorld_TASK_TYPE["tactile"]["task_ids"]
        tasks = [t for t in tasks if t[0].stem not in remaining_ids]
    else:
        tasks = [t for t in tasks if t[0].stem in type_task_ids]

    if task_ids:
        tasks = [t for t in tasks if t[0].stem in task_ids]

    print(f"Tasks to process: {len(tasks)}", flush=True)

    if debug:
        save_as_lerobot_dataset_fast(config, tasks[0], skip_video_stats, staging_root=staging_root)
    else:
        runtime_env = RuntimeEnv(
            env_vars={"HDF5_USE_FILE_LOCKING": "FALSE", "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE"}
        )
        ray.init(runtime_env=runtime_env)
        cpus = int(ray.available_resources()["CPU"])
        n_workers = cpus // cpus_per_task
        print(
            f"Available CPUs: {cpus}, cpus_per_task: {cpus_per_task}, "
            f"workers: {n_workers}, max_tar_readers: {max_tar_readers}",
            flush=True,
        )

        tar_sem = TarSemaphore.remote(max_tar_readers)

        remote_fn = ray.remote(save_as_lerobot_dataset_fast).options(num_cpus=cpus_per_task)
        futures = [
            (t[0].stem, remote_fn.remote(config, t, skip_video_stats, staging_root=staging_root, tar_sem=tar_sem))
            for t in tasks
        ]

        for task_name, future in futures:
            try:
                ray.get(future)
            except Exception as e:
                print(f"FAILED {task_name}: {e}", flush=True)
                with open("output_fast.txt", "a") as f:
                    f.write(f"{task_name}: {e}\n")

        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast AgiBotWorld → LeRobot v2.1 converter")
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--eef-type", type=str, choices=["gripper", "dexhand", "tactile"], default="gripper")
    parser.add_argument("--task-ids", type=str, nargs="+", help="task_327 task_351 ...", default=[])
    parser.add_argument("--cpus-per-task", type=int, default=3)
    parser.add_argument("--skip-video-stats", action="store_true", help="Skip video frame decoding for stats (faster)")
    parser.add_argument("--staging-dir", type=str, default=None, help="Local disk path for staging (default: /tmp)")
    parser.add_argument("--max-tar-readers", type=int, default=DEFAULT_MAX_TAR_READERS,
                        help=f"Max concurrent NFS tar reads (default: {DEFAULT_MAX_TAR_READERS})")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
