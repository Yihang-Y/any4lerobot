#!/usr/bin/env python
"""
Post-process DROID LeRobot v2.1 dataset:
  1. Build (lab, date, time) index from droid_language_annotations.json
  2. Build rel_path index from keep_ranges_1_0_1.json
  3. For each episode (via droid_mapping.jsonl):
     - Match keep_ranges (rel_path direct match)
     - Match language annotation ((lab, date, time) match, ignore 206 collisions)
  4. Update meta files IN-PLACE:
     - tasks.jsonl: append new task strings
     - episodes.jsonl: update tasks list (all 3 instructions)
     - info.json: update total_tasks
     - NEW keep_ranges.jsonl: episode_index, is_success, keep_ranges, rel_path
  5. Update parquet files: rewrite task_index column for annotated episodes

Usage:
    python postprocess_droid.py \
        --mapping /home/fox/vla_linear/any4lerobot/info/droid_mapping.jsonl \
        --lang-ann /home/fox/vla_linear/any4lerobot/info/droid_language_annotations.json \
        --keep-ranges /home/fox/vla_linear/any4lerobot/info/keep_ranges_1_0_1.json \
        --lerobot-dir /tmp/droid_lerobot/droid_1.0.0_lerobot
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq


# ── helpers ───────────────────────────────────────────────────────────


def extract_rel_path(gcs_path: str) -> str:
    """
    Extract relative path from GCS URL.
    "gs://xembodiment_data/r2d2/r2d2-data-full/BVL/success/2024-02-02/..."
    -> "BVL/success/2024-02-02/Fri_Feb__2_19:55:01_2024"
    """
    prefix = "gs://xembodiment_data/r2d2/r2d2-data-full/"
    if gcs_path.startswith(prefix):
        rel = gcs_path[len(prefix):]
    else:
        rel = gcs_path
    # Remove trailing /recordings/MP4 or similar
    rel = re.sub(r"/recordings.*$", "", rel)
    rel = re.sub(r"/trajectory\.h5$", "", rel)
    rel = rel.rstrip("/")
    return rel


def extract_lab_date_time(s: str):
    """
    From a rel_path like "BVL/success/2024-02-02/Fri_Feb__2_19:55:01_2024"
    or an episode_id like "AUTOLab+5d05c5aa+2023-07-07-09h-45m-39s"
    returns (lab, date_str, time_str) e.g. ("BVL", "2024-02-02", "19:55:01")
    """
    # Try rel_path format (slash-separated)
    parts = s.split("/")
    if len(parts) >= 4:
        lab = parts[0]
        date = parts[2]  # YYYY-MM-DD
        time_part = parts[3]  # e.g. "Fri_Feb__2_19:55:01_2024"
        m = re.search(r"(\d{2}:\d{2}:\d{2})", time_part)
        if m:
            return (lab, date, m.group(1))

    # Try episode_id format (plus-separated)
    parts = s.split("+")
    if len(parts) >= 3:
        lab = parts[0]
        dt = parts[-1]  # e.g. "2023-07-07-09h-45m-39s"
        dm = re.match(r"(\d{4}-\d{2}-\d{2})", dt)
        tm = re.search(r"(\d{2})h-(\d{2})m-(\d{2})s", dt)
        if dm and tm:
            date = dm.group(1)
            time_str = f"{tm.group(1)}:{tm.group(2)}:{tm.group(3)}"
            return (lab, date, time_str)

    return None


# ── index builders ────────────────────────────────────────────────────


def build_keep_ranges_index(keep_ranges_path: Path) -> dict:
    """
    Returns dict: rel_path -> [[start, end], ...]
    Key is the relative path extracted from the GCS compound key.
    """
    print("Building keep_ranges index...", flush=True)
    index = {}
    with open(keep_ranges_path) as f:
        data = json.load(f)
    for compound_key, ranges in data.items():
        # compound key: "gs://.../MP4--gs://.../trajectory.h5"
        parts = compound_key.split("--")
        h5_path = parts[1] if len(parts) > 1 else parts[0]
        rel = extract_rel_path(h5_path)
        index[rel] = ranges
    print(f"  {len(index)} entries in keep_ranges index", flush=True)
    return index


def build_lang_ann_index(lang_ann_path: Path):
    """
    Returns:
      - exact_index: rel_path (from episode_id_to_path) -> {lang1, lang2, lang3}
        Not used since we don't have episode_id_to_path, but kept for reference.
      - timestamp_index: (lab, date, time) -> [episode_id, ...]
        If only 1 entry per key: unambiguous match. Collisions (len>1): skip.
    """
    print("Building language annotation index...", flush=True)
    with open(lang_ann_path) as f:
        data = json.load(f)

    timestamp_index = defaultdict(list)
    for ep_id, ann in data.items():
        key = extract_lab_date_time(ep_id)
        if key:
            timestamp_index[key].append(ep_id)

    # Identify collisions
    n_collision = sum(1 for v in timestamp_index.values() if len(v) > 1)
    n_unique = sum(1 for v in timestamp_index.values() if len(v) == 1)
    print(f"  {len(data)} total annotations", flush=True)
    print(f"  {n_unique} unique (lab,date,time) keys, {n_collision} collisions (will be ignored)", flush=True)

    # Build clean index: only keep unambiguous entries
    clean_index = {}
    for key, ep_ids in timestamp_index.items():
        if len(ep_ids) == 1:
            ep_id = ep_ids[0]
            clean_index[key] = data[ep_id]

    return clean_index


def load_mapping(mapping_path: Path) -> list:
    print(f"Loading mapping from {mapping_path}...", flush=True)
    entries = []
    with open(mapping_path) as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"  {len(entries)} episodes in mapping", flush=True)
    return entries


# ── main processing ───────────────────────────────────────────────────


def process_dataset(
    mapping_path: Path,
    lang_ann_path: Path,
    keep_ranges_path: Path,
    lerobot_dir: Path,
):
    t0 = time.time()

    keep_index = build_keep_ranges_index(keep_ranges_path)
    lang_index = build_lang_ann_index(lang_ann_path)
    mapping = load_mapping(mapping_path)

    print(f"\nProcessing {len(mapping)} episodes...", flush=True)

    meta_dir = lerobot_dir / "meta"

    # Load existing tasks
    existing_tasks = {}  # task_index -> task_str
    existing_task_to_idx = {}  # task_str -> task_index
    with open(meta_dir / "tasks.jsonl") as f:
        for line in f:
            t = json.loads(line)
            existing_tasks[t["task_index"]] = t["task"]
            existing_task_to_idx[t["task"]] = t["task_index"]

    next_task_idx = max(existing_tasks.keys()) + 1
    print(f"Existing tasks: {len(existing_tasks)} (next_idx={next_task_idx})", flush=True)

    # Load existing episodes
    existing_episodes = {}  # episode_index -> {tasks, length}
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            existing_episodes[ep["episode_index"]] = ep

    # Prepare per-episode updates
    # ep_idx -> {new_tasks: [str,...], new_task_index: int, keep_ranges, is_success, rel_path}
    updates = {}
    new_tasks_set = {}  # task_str -> task_index (accumulates new tasks)

    n_matched_lang = 0
    n_matched_keep = 0
    n_success = 0
    n_failure = 0
    n_not_in_keep = 0

    for entry in mapping:
        ep_idx = entry["lerobot_idx"]
        recording_fp = entry.get("recording_folderpath", "")

        rel_path = extract_rel_path(recording_fp) if recording_fp else ""
        is_success = "/success/" in rel_path
        if is_success:
            n_success += 1
        else:
            n_failure += 1

        # Match keep_ranges
        keep_ranges = keep_index.get(rel_path)
        if keep_ranges is not None:
            n_matched_keep += 1

        # Match language annotation via (lab, date, time)
        lang_ann = None
        if rel_path:
            key = extract_lab_date_time(rel_path)
            if key:
                lang_ann = lang_index.get(key)

        # Merge JSON annotations with existing RLDS task (dedup, JSON first)
        existing_ep = existing_episodes.get(ep_idx, {})
        existing_tasks = existing_ep.get("tasks", [])

        seen = set()
        merged_tasks = []
        if lang_ann is not None:
            n_matched_lang += 1
            for key_name in ["language_instruction1", "language_instruction2", "language_instruction3"]:
                t = (lang_ann.get(key_name) or "").strip()
                if t and t not in seen:
                    seen.add(t)
                    merged_tasks.append(t)
        for t in existing_tasks:
            if t and t not in seen:
                seen.add(t)
                merged_tasks.append(t)

        new_tasks_list = merged_tasks if merged_tasks else None

        if new_tasks_list:
            primary_task = new_tasks_list[0]
            for t in new_tasks_list:
                if t not in existing_task_to_idx and t not in new_tasks_set:
                    new_tasks_set[t] = next_task_idx
                    next_task_idx += 1
            new_task_index = (
                existing_task_to_idx.get(primary_task)
                or new_tasks_set.get(primary_task)
            )
        else:
            new_task_index = None

        if keep_ranges is None:
            n_not_in_keep += 1

        updates[ep_idx] = {
            "new_tasks": new_tasks_list,
            "new_task_index": new_task_index,
            "keep_ranges": keep_ranges,
            "is_success": is_success,
            "rel_path": rel_path,
        }

    print(f"\nMatching results:", flush=True)
    print(f"  Success episodes: {n_success}", flush=True)
    print(f"  Failure episodes: {n_failure}", flush=True)
    print(f"  Matched keep_ranges: {n_matched_keep}", flush=True)
    print(f"  Not in keep_ranges: {n_not_in_keep}", flush=True)
    print(f"  Matched language annotation: {n_matched_lang}", flush=True)
    print(f"  New tasks to add: {len(new_tasks_set)}", flush=True)

    # ── Write updated meta files ───────────────────────────────────────

    print("\nWriting meta files...", flush=True)

    # 1. Append new tasks to tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "a") as f:
        for task_str, task_idx in sorted(new_tasks_set.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": task_idx, "task": task_str}) + "\n")
    total_tasks = len(existing_tasks) + len(new_tasks_set)
    print(f"  tasks.jsonl: appended {len(new_tasks_set)} new tasks (total={total_tasks})", flush=True)

    # 2. Rewrite episodes.jsonl with updated tasks
    all_task_to_idx = {**existing_task_to_idx, **new_tasks_set}
    updated_episodes = []
    for ep_idx in sorted(existing_episodes.keys()):
        ep = existing_episodes[ep_idx]
        upd = updates.get(ep_idx, {})
        new_tasks = upd.get("new_tasks")
        if new_tasks is not None:
            ep = dict(ep)
            ep["tasks"] = new_tasks
            ep["task_index"] = upd.get("new_task_index", ep.get("task_index"))
        updated_episodes.append(ep)

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in updated_episodes:
            f.write(json.dumps(ep) + "\n")
    print(f"  episodes.jsonl: rewritten with updated tasks", flush=True)

    # 3. Update info.json
    with open(meta_dir / "info.json") as f:
        info = json.load(f)
    info["total_tasks"] = total_tasks
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    print(f"  info.json: total_tasks updated to {total_tasks}", flush=True)

    # 4. Write keep_ranges.jsonl
    with open(meta_dir / "keep_ranges.jsonl", "w") as f:
        for ep_idx in sorted(updates.keys()):
            upd = updates[ep_idx]
            f.write(json.dumps({
                "episode_index": ep_idx,
                "is_success": upd["is_success"],
                "keep_ranges": upd["keep_ranges"],
                "rel_path": upd["rel_path"],
            }) + "\n")
    print(f"  keep_ranges.jsonl: written {len(updates)} entries", flush=True)

    # ── Rewrite parquet files ──────────────────────────────────────────

    print("\nRewriting parquet files...", flush=True)

    # Collect episodes that need parquet updates (those with new_task_index)
    parquet_updates = {
        ep_idx: upd["new_task_index"]
        for ep_idx, upd in updates.items()
        if upd["new_task_index"] is not None
    }
    print(f"  Episodes needing parquet update: {len(parquet_updates)}", flush=True)

    def rewrite_parquet(ep_idx, new_task_idx):
        chunk = ep_idx // 1000
        path = lerobot_dir / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        if not path.exists():
            return ep_idx, False, "not found"
        try:
            table = pq.read_table(path)
            n_rows = len(table)
            new_task_col = pa.array([new_task_idx] * n_rows, type=pa.int64())
            # Replace task_index column
            col_names = table.schema.names
            if "task_index" not in col_names:
                return ep_idx, False, "no task_index column"
            idx = col_names.index("task_index")
            arrays = [table.column(i) for i in range(len(col_names))]
            arrays[idx] = new_task_col
            new_table = pa.table(dict(zip(col_names, arrays)))
            pq.write_table(new_table, path)
            return ep_idx, True, None
        except Exception as e:
            return ep_idx, False, str(e)

    t_parquet = time.time()
    n_done = 0
    n_failed = 0
    total = len(parquet_updates)

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(rewrite_parquet, ep_idx, new_task_idx): ep_idx
            for ep_idx, new_task_idx in parquet_updates.items()
        }
        for fut in as_completed(futures):
            ep_idx, ok, err = fut.result()
            if ok:
                n_done += 1
            else:
                n_failed += 1
                if n_failed <= 10:
                    print(f"  WARNING: episode {ep_idx} parquet failed: {err}", flush=True)
            if (n_done + n_failed) % 5000 == 0 and (n_done + n_failed) > 0:
                elapsed = time.time() - t_parquet
                rate = (n_done + n_failed) / elapsed
                print(
                    f"  Parquet: {n_done+n_failed}/{total} done | {rate:.0f} ep/s | {n_failed} failed",
                    flush=True,
                )

    elapsed_parquet = time.time() - t_parquet
    print(f"  Parquet rewrite done: {n_done} ok, {n_failed} failed in {elapsed_parquet:.0f}s", flush=True)

    total_elapsed = time.time() - t0
    print(f"\nAll done in {total_elapsed:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True,
                        help="Path to droid_mapping.jsonl")
    parser.add_argument("--lang-ann", type=Path, required=True,
                        help="Path to droid_language_annotations.json")
    parser.add_argument("--keep-ranges", type=Path, required=True,
                        help="Path to keep_ranges_1_0_1.json")
    parser.add_argument("--lerobot-dir", type=Path, required=True,
                        help="Path to LeRobot dataset directory")
    args = parser.parse_args()

    process_dataset(
        mapping_path=args.mapping,
        lang_ann_path=args.lang_ann,
        keep_ranges_path=args.keep_ranges,
        lerobot_dir=args.lerobot_dir,
    )


if __name__ == "__main__":
    main()
