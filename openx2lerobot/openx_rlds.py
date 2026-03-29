#!/usr/bin/env python
"""
Fast converter: Open X-Embodiment RLDS → LeRobot v2.1 format.

Bypasses LeRobot's add_frame/save_episode to avoid PNG encode/decode round-trips.
Writes parquet, mp4 (via PyAV pipe), and meta files directly.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

os.environ["SVT_LOG"] = "0"

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
import tensorflow_datasets as tfds
from oxe_utils.configs import OXE_DATASET_CONFIGS, ActionEncoding, StateEncoding
from oxe_utils.transforms import OXE_STANDARDIZATION_TRANSFORMS

np.set_printoptions(precision=2)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
QUANTILE_KEYS = ["q01", "q10", "q50", "q90", "q99"]
PROGRESS_FILE = "meta/progress.jsonl"
CHECKPOINT_FILE = "meta/checkpoint.json"


# ── checkpoint / resume ───────────────────────────────────────────────


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def _save_progress_line(local_dir, line_dict):
    path = local_dir / PROGRESS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(line_dict, default=_json_default) + "\n")


def _save_checkpoint(local_dir, completed_episodes, global_frame_idx, task_to_idx):
    ckpt = {
        "completed_episodes": completed_episodes,
        "global_frame_idx": global_frame_idx,
        "task_to_idx": task_to_idx,
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
        return 0, None

    with open(ckpt_path) as f:
        ckpt = json.load(f)
    n = ckpt["completed_episodes"]
    if n == 0:
        return 0, None

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

    # Truncate progress to checkpoint boundary
    with open(prog_path, "w") as f:
        for entry in all_progress:
            f.write(json.dumps(entry, default=_json_default) + "\n")

    # Reconstruct all_episode_stats from progress
    all_stats = []
    for entry in all_progress:
        stats = entry.get("stats", {})
        np_stats = {}
        for feat_key, feat_stats in stats.items():
            np_stats[feat_key] = {k: np.array(v) for k, v in feat_stats.items()}
        all_stats.append(np_stats)

    return n, {
        "all_progress": all_progress,
        "all_episode_stats": all_stats,
        "global_frame_idx": ckpt["global_frame_idx"],
        "task_to_idx": ckpt["task_to_idx"],
    }


# ── transform (unchanged) ─────────────────────────────────────────────


def transform_raw_dataset(episode, dataset_name):
    traj = next(iter(episode["steps"].batch(episode["steps"].cardinality())))

    if dataset_name in OXE_STANDARDIZATION_TRANSFORMS:
        traj = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](traj)

    if dataset_name in OXE_DATASET_CONFIGS:
        state_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["state_obs_keys"]
    else:
        state_obs_keys = [None for _ in range(8)]

    proprio = tf.concat(
        [
            (
                tf.zeros((tf.shape(traj["action"])[0], 1), dtype=tf.float32)
                if key is None
                else tf.cast(traj["observation"][key], tf.float32)
            )
            for key in state_obs_keys
        ],
        axis=1,
    )

    traj.update(
        {
            "proprio": proprio,
            "task": traj.pop("language_instruction"),
            "action": tf.cast(traj["action"], tf.float32),
        }
    )

    episode["steps"] = traj
    return episode


# ── feature / name helpers ────────────────────────────────────────────


def _get_state_action_names(dataset_name):
    state_names = [f"motor_{i}" for i in range(8)]
    action_names = [f"motor_{i}" for i in range(8)]

    if dataset_name in OXE_DATASET_CONFIGS:
        cfg = OXE_DATASET_CONFIGS[dataset_name]
        se = cfg["state_encoding"]
        if se == StateEncoding.POS_EULER:
            state_names = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
            if "libero" in dataset_name:
                state_names = ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper"]
        elif se == StateEncoding.POS_QUAT:
            state_names = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
        elif se == StateEncoding.JOINT:
            state_names = [f"motor_{i}" for i in range(7)] + ["gripper"]
            skeys = cfg["state_obs_keys"]
            pad_count = skeys[:-1].count(None)
            state_names[-pad_count - 1 : -1] = ["pad"] * pad_count
            state_names[-1] = "pad" if skeys[-1] is None else state_names[-1]

        ae = cfg["action_encoding"]
        if ae == ActionEncoding.EEF_POS:
            action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            if "libero" in dataset_name:
                action_names = ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]
        elif ae == ActionEncoding.JOINT_POS:
            action_names = [f"motor_{i}" for i in range(7)] + ["gripper"]

    return state_names, action_names


def _get_image_keys(builder):
    obs = builder.info.features["steps"]["observation"]
    return [
        key
        for key, value in obs.items()
        if "depth" not in key and any(x in key for x in ["image", "rgb"])
    ]


# ── stats helpers ─────────────────────────────────────────────────────


def _compute_array_stats(arr):
    return {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "count": np.array([len(arr)]),
    }


def _compute_image_stats(frames, sample_n=10, spatial_stride=4):
    n = len(frames)
    indices = np.linspace(0, n - 1, min(sample_n, n)).astype(int)
    sampled = frames[indices, ::spatial_stride, ::spatial_stride, :]
    flat = sampled.reshape(-1, 3).astype(np.float32) * (1.0 / 255.0)
    ch_min = flat.min(axis=0).reshape(3, 1, 1)
    ch_max = flat.max(axis=0).reshape(3, 1, 1)
    ch_mean = flat.mean(axis=0).reshape(3, 1, 1)
    ch_std = flat.std(axis=0).reshape(3, 1, 1)
    result = {
        "min": ch_min, "max": ch_max, "mean": ch_mean, "std": ch_std,
        "count": np.array([len(sampled)]),
    }
    return result


def _stats_to_jsonable(stats):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()}


# ── video encoding via PyAV ──────────────────────────────────────────


def _encode_video_pyav(frames, output_path, fps, vcodec="libsvtav1", crf=30, pix_fmt="yuv420p"):
    n, h, w, _ = frames.shape
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream(vcodec, rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = pix_fmt
    if crf is not None:
        stream.options["crf"] = str(crf)
    if vcodec == "libsvtav1":
        stream.options["preset"] = "10"
    stream.options["g"] = "2"

    for i in range(n):
        frame = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


# ── safe iterator ─────────────────────────────────────────────────────


def _iter_episodes_safe(raw_dataset):
    it = raw_dataset.as_numpy_iterator()
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except tf.errors.DataLossError as e:
            print(f"WARNING: skipping corrupted record: {e}", flush=True)
            continue


# ── per-episode parquet writer (v2.1) ─────────────────────────────────


def _write_episode_parquet(local_dir, ep_idx, ep_data):
    """Write one parquet file per episode: data/chunk-{chunk}/episode_{idx:06d}.parquet"""
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
            arrays[key] = pa.FixedSizeListArray.from_arrays(
                pa.array(arr.reshape(-1).astype(np.float32)),
                list_size=arr.shape[1],
            )
    table = pa.table(arrays)
    out_path = local_dir / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


# ── meta writers (v2.1) ──────────────────────────────────────────────


def _write_episodes_jsonl(local_dir, all_progress):
    out_path = local_dir / "meta/episodes.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for entry in all_progress:
            line = {
                "episode_index": entry["episode_index"],
                "tasks": entry["tasks"],
                "length": entry["length"],
            }
            f.write(json.dumps(line) + "\n")


def _write_episodes_stats_jsonl(local_dir, all_progress):
    out_path = local_dir / "meta/episodes_stats.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for entry in all_progress:
            line = {
                "episode_index": entry["episode_index"],
                "stats": entry["stats"],
            }
            f.write(json.dumps(line, default=_json_default) + "\n")


def _write_tasks_jsonl(local_dir, task_to_idx):
    sorted_tasks = sorted(task_to_idx.items(), key=lambda x: x[1])
    out_path = local_dir / "meta/tasks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for task, idx in sorted_tasks:
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")


def _write_keep_ranges_jsonl(local_dir, all_progress):
    """Write meta/keep_ranges.jsonl from all_progress (when DROID postprocess was used)."""
    out_path = local_dir / "meta/keep_ranges.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for entry in all_progress:
            line = {
                "episode_index": entry["episode_index"],
                "is_success": entry.get("is_success", False),
                "keep_ranges": entry.get("keep_ranges"),
                "rel_path": entry.get("rel_path", ""),
            }
            f.write(json.dumps(line) + "\n")
    print(f"  keep_ranges.jsonl: written {len(all_progress)} entries", flush=True)


def _write_info_json(
    local_dir, dataset_name, robot_type, fps, use_videos,
    total_episodes, total_frames, total_tasks,
    state_names, action_names, image_keys, builder,
):
    obs = builder.info.features["steps"]["observation"]
    features = {}

    total_videos = 0
    for img_key in image_keys:
        shape = list(obs[img_key].shape)
        feat = {
            "dtype": "video" if use_videos else "image",
            "shape": shape,
            "names": ["height", "width", "rgb"],
        }
        if use_videos:
            feat["info"] = {
                "video.height": shape[0],
                "video.width": shape[1],
                "video.codec": "av1" if "svt" in "libsvtav1" else "h264",
                "video.pix_fmt": "yuv420p",
                "video.fps": fps,
                "video.channels": shape[2],
            }
            total_videos += total_episodes
        features[f"observation.images.{img_key}"] = feat

    features["observation.state"] = {
        "dtype": "float32",
        "shape": [len(state_names)],
        "names": {"motors": state_names},
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [len(action_names)],
        "names": {"motors": action_names},
    }
    for sys_key, dtype in [("timestamp", "float32"), ("frame_index", "int64"),
                            ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")]:
        features[sys_key] = {"dtype": dtype, "shape": [1], "names": None}

    total_chunks = (total_episodes + DEFAULT_CHUNK_SIZE - 1) // DEFAULT_CHUNK_SIZE

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4" if use_videos else None,
        "features": features,
    }

    out_path = local_dir / "meta/info.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=4)


def _write_stats_json(local_dir, all_episode_stats):
    if not all_episode_stats:
        return
    all_keys = list(all_episode_stats[0].keys())
    global_stats = {}

    for key in all_keys:
        ep_stats_list = [ep[key] for ep in all_episode_stats]
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

    out_path = local_dir / "meta/stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(global_stats, f, indent=4)


# ── main conversion ───────────────────────────────────────────────────


def _load_droid_postprocess_indexes(lang_ann_path, keep_ranges_path):
    """Load language annotation and keep_ranges indexes for DROID (used during conversion)."""
    from postprocess_droid import build_keep_ranges_index, build_lang_ann_index
    lang_index = build_lang_ann_index(Path(lang_ann_path))
    keep_index = build_keep_ranges_index(Path(keep_ranges_path))
    return lang_index, keep_index


def _droid_episode_tasks_and_keep(episode_metadata, lang_index, keep_index):
    """
    Given episode_metadata (dict with recording_folderpath, pre-decoded to str),
    return (tasks_list, keep_ranges, rel_path, is_success).
    If no lang match, tasks_list is None and caller uses RLDS task for task_str/task_index.
    """
    from postprocess_droid import extract_rel_path, extract_lab_date_time

    recording_fp = episode_metadata.get("recording_folderpath") or ""
    rel_path = extract_rel_path(recording_fp) if recording_fp else ""
    is_success = "/success/" in rel_path
    keep_ranges = keep_index.get(rel_path)

    key = extract_lab_date_time(rel_path) if rel_path else None
    lang_ann = lang_index.get(key) if key else None
    if lang_ann is None:
        return None, keep_ranges, rel_path, is_success

    instr1 = (lang_ann.get("language_instruction1") or "").strip()
    instr2 = (lang_ann.get("language_instruction2") or "").strip()
    instr3 = (lang_ann.get("language_instruction3") or "").strip()
    seen = set()
    new_tasks_list = []
    for t in [instr1, instr2, instr3]:
        if t and t not in seen:
            seen.add(t)
            new_tasks_list.append(t)
    return new_tasks_list, keep_ranges, rel_path, is_success


def create_lerobot_dataset(
    raw_dir: Path,
    local_dir: Path,
    robot_type: str = None,
    fps: int = None,
    use_videos: bool = False,
    vcodec: str = "libsvtav1",
    repo_id: str = None,
    push_to_hub: bool = False,
    lang_ann: Path = None,
    keep_ranges: Path = None,
    **kwargs,
):
    last_part = raw_dir.name
    if re.match(r"^\d+\.\d+\.\d+$", last_part):
        version = last_part
        dataset_name = raw_dir.parent.name
        data_dir = raw_dir.parent.parent
    else:
        version = ""
        dataset_name = last_part
        data_dir = raw_dir.parent

    local_dir = Path(local_dir) / f"{dataset_name}_{version}_lerobot"
    local_dir.mkdir(parents=True, exist_ok=True)

    # DROID: optional in-conversion postprocess (language annotations + keep_ranges)
    droid_postprocess = (
        dataset_name == "droid"
        and lang_ann is not None
        and keep_ranges is not None
        and Path(lang_ann).exists()
        and Path(keep_ranges).exists()
    )
    lang_index = None
    keep_index = None
    if droid_postprocess:
        print("DROID postprocess enabled: language annotations + keep_ranges applied during conversion.", flush=True)
        lang_index, keep_index = _load_droid_postprocess_indexes(lang_ann, keep_ranges)

    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
    image_keys = _get_image_keys(builder)
    state_names, action_names = _get_state_action_names(dataset_name)

    if fps is None:
        fps = OXE_DATASET_CONFIGS.get(dataset_name, {}).get("control_frequency", 10)
    if robot_type is None:
        rt = OXE_DATASET_CONFIGS.get(dataset_name, {}).get("robot_type", "unknown")
        robot_type = rt.lower().replace(" ", "_").replace("-", "_")

    # Build tf.data pipeline
    filter_fn = lambda e: e["success"] if dataset_name == "kuka" else True
    read_config = tfds.ReadConfig(
        interleave_cycle_length=64,
        interleave_block_length=1,
        num_parallel_calls_for_interleave_files=tf.data.AUTOTUNE,
    )
    try:
        total_episodes = builder.info.splits["train"].num_examples
    except Exception:
        total_episodes = None
    total_str = str(total_episodes) if total_episodes else "?"

    # Try to resume from checkpoint
    n_skip, resume_state = _load_resume_state(local_dir)
    if resume_state:
        global_frame_idx = resume_state["global_frame_idx"]
        task_to_idx = resume_state["task_to_idx"]
        all_progress = resume_state["all_progress"]
        all_episode_stats = resume_state["all_episode_stats"]
        print(f"Resuming from episode {n_skip} ({global_frame_idx} frames done)", flush=True)
    else:
        global_frame_idx = 0
        task_to_idx = {}
        all_progress = []
        all_episode_stats = []

    filtered_dataset = builder.as_dataset(split="train", read_config=read_config).filter(filter_fn)
    if n_skip > 0:
        filtered_dataset = filtered_dataset.skip(n_skip)
    raw_dataset = (
        filtered_dataset
        .map(partial(transform_raw_dataset, dataset_name=dataset_name), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    n_video_workers = max(len(image_keys), 1) if use_videos else 1
    video_pool = ThreadPoolExecutor(max_workers=n_video_workers)

    t0 = time.time()
    t_prev = t0

    for i, episode in enumerate(_iter_episodes_safe(raw_dataset)):
        ep_idx = n_skip + i
        t_ep = time.time()
        traj = episode["steps"]
        n_frames = traj["action"].shape[0]

        if droid_postprocess:
            meta = episode.get("episode_metadata") or {}
            if isinstance(meta, dict):
                meta = {k: (v.decode("utf-8", errors="replace") if hasattr(v, "decode") else v) for k, v in meta.items()}
            else:
                meta = {}
            droid_tasks, ep_keep_ranges, rel_path, is_success = _droid_episode_tasks_and_keep(meta, lang_index, keep_index)

            rlds_task = traj["task"][0]
            if hasattr(rlds_task, "decode"):
                rlds_task = rlds_task.decode()
            else:
                rlds_task = str(rlds_task)

            seen = set()
            tasks_for_episode = []
            for t in (droid_tasks or []):
                if t and t not in seen:
                    seen.add(t)
                    tasks_for_episode.append(t)
            if rlds_task and rlds_task not in seen:
                tasks_for_episode.append(rlds_task)

            for t in tasks_for_episode:
                if t not in task_to_idx:
                    task_to_idx[t] = len(task_to_idx)
            task_idx = task_to_idx[tasks_for_episode[0]]
        else:
            task_str = traj["task"][0].decode()
            if task_str not in task_to_idx:
                task_to_idx[task_str] = len(task_to_idx)
            task_idx = task_to_idx[task_str]
            tasks_for_episode = [task_str]
            ep_keep_ranges, rel_path, is_success = None, "", False

        ep_chunk = ep_idx // DEFAULT_CHUNK_SIZE

        # Per-frame data
        timestamps = np.arange(n_frames, dtype=np.float32) / fps
        frame_indices = np.arange(n_frames, dtype=np.int64)
        episode_indices = np.full(n_frames, ep_idx, dtype=np.int64)
        global_indices = np.arange(global_frame_idx, global_frame_idx + n_frames, dtype=np.int64)
        task_indices = np.full(n_frames, task_idx, dtype=np.int64)

        ep_data = {
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": episode_indices,
            "index": global_indices,
            "task_index": task_indices,
            "observation.state": traj["proprio"].astype(np.float32),
            "action": traj["action"].astype(np.float32),
        }

        # Write per-episode parquet immediately
        _write_episode_parquet(local_dir, ep_idx, ep_data)

        # Compute stats
        ep_stats = {}
        ep_stats["observation.state"] = _compute_array_stats(traj["proprio"].astype(np.float32))
        ep_stats["action"] = _compute_array_stats(traj["action"].astype(np.float32))
        ep_stats["timestamp"] = _compute_array_stats(timestamps.reshape(-1, 1))
        ep_stats["frame_index"] = _compute_array_stats(frame_indices.astype(np.float64).reshape(-1, 1))
        ep_stats["episode_index"] = _compute_array_stats(episode_indices.astype(np.float64).reshape(-1, 1))
        ep_stats["index"] = _compute_array_stats(global_indices.astype(np.float64).reshape(-1, 1))
        ep_stats["task_index"] = _compute_array_stats(task_indices.astype(np.float64).reshape(-1, 1))

        t_stats = time.time()

        # Submit video encoding, then compute image stats in parallel
        video_futures = []
        image_frames = {}
        for img_key in image_keys:
            frames = traj["observation"][img_key]
            image_frames[img_key] = frames

            if use_videos:
                vid_key = f"observation.images.{img_key}"
                vid_path = local_dir / f"videos/chunk-{ep_chunk:03d}/{vid_key}/episode_{ep_idx:06d}.mp4"
                video_futures.append(
                    video_pool.submit(_encode_video_pyav, frames, vid_path, fps, vcodec)
                )

        for img_key, frames in image_frames.items():
            ep_stats[f"observation.images.{img_key}"] = _compute_image_stats(frames)

        t_vid_submit = time.time()

        for fut in video_futures:
            fut.result()

        t_vid_done = time.time()

        all_episode_stats.append(ep_stats)

        # Build progress entry (includes stats for episodes_stats.jsonl)
        progress_entry = {
            "episode_index": ep_idx,
            "tasks": tasks_for_episode,
            "length": n_frames,
            "stats": {k: _stats_to_jsonable(v) for k, v in ep_stats.items()},
        }
        if droid_postprocess:
            progress_entry["keep_ranges"] = ep_keep_ranges
            progress_entry["rel_path"] = rel_path
            progress_entry["is_success"] = is_success
        all_progress.append(progress_entry)
        global_frame_idx += n_frames
        _save_progress_line(local_dir, progress_entry)

        # Checkpoint every 10 episodes
        if (i + 1) % 10 == 0:
            _save_checkpoint(local_dir, ep_idx + 1, global_frame_idx, task_to_idx)

        elapsed = time.time() - t0
        ep_time = time.time() - t_ep
        eps_this_run = i + 1
        eps_per_sec = eps_this_run / elapsed if elapsed > 0 else 0
        print(
            f"Episode {ep_idx}/{total_str} | {n_frames}f | "
            f"iter={t_ep - (t0 if i == 0 else t_prev):.1f}s "
            f"stats={t_vid_submit - t_stats:.1f}s "
            f"vid={t_vid_done - t_vid_submit:.1f}s | "
            f"{ep_time:.1f}s | {eps_per_sec:.2f} ep/s | {elapsed:.0f}s",
            flush=True,
        )
        t_prev = time.time()

    video_pool.shutdown(wait=True)

    # Write final meta files
    _write_episodes_jsonl(local_dir, all_progress)
    _write_episodes_stats_jsonl(local_dir, all_progress)
    _write_tasks_jsonl(local_dir, task_to_idx)
    if droid_postprocess:
        _write_keep_ranges_jsonl(local_dir, all_progress)
    _write_info_json(
        local_dir, dataset_name, robot_type, fps, use_videos,
        len(all_progress), global_frame_idx, len(task_to_idx),
        state_names, action_names, image_keys, builder,
    )
    _write_stats_json(local_dir, all_episode_stats)
    _save_checkpoint(local_dir, len(all_progress), global_frame_idx, task_to_idx)

    print(f"\nDone! {len(all_progress)} episodes, {global_frame_idx} frames → {local_dir}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Fast RLDS → LeRobot v2.1 converter")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--robot-type", type=str, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--vcodec", type=str, default="libsvtav1")
    parser.add_argument("--lang-ann", type=Path, default=None,
                        help="DROID: path to droid_language_annotations.json (enables in-conversion postprocess)")
    parser.add_argument("--keep-ranges", type=Path, default=None,
                        help="DROID: path to keep_ranges_1_0_1.json (enables in-conversion postprocess)")
    args = parser.parse_args()
    create_lerobot_dataset(**vars(args))


if __name__ == "__main__":
    main()
