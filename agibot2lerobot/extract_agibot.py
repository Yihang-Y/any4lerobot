#!/usr/bin/env python
"""
解压 AgiBot-World 原始数据中的 tar 包，得到 observations/{task_id}/{episode_id}/ 和
proprio_stats/{task_id}/{episode_id}/ 结构，供 agibot_fast.py / agibot_h5.py 使用。

用法:
  python extract_agibot.py --root /path/to/OpenDriveLab___AgiBot-World/raw/main
  python extract_agibot.py --root /path/to/raw/main --jobs 8   # 并行解压
  python extract_agibot.py --root /path/to/raw/main --dry-run  # 只列将要解压的 tar
  python extract_agibot.py --root /path/to/raw/main --resume  # 断点续传，跳过已解压的 tar
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 进度条宽度（字符数）
PROGRESS_BAR_WIDTH = 40
# 进度条刷新间隔（秒）
PROGRESS_REFRESH_INTERVAL = 1.0
# 解压完成后写入的标记文件，用于断点续传时区分「完整解压」与「解压到一半中断」
EXTRACTED_MARKER = ".extracted"


def _tar_first_top_dir(tar_path: Path) -> str | None:
    """只读 tar 的第一条记录头，取顶层目录名。不遍历整个 tar。"""
    try:
        with tarfile.open(tar_path, "r") as tf:
            first = next(iter(tf), None)
            if first is None:
                return None
            return first.name.split("/")[0] if first.name else None
    except Exception:
        return None


def extract_one(
    tar_path: Path, dest_dir: Path, dry_run: bool, skip_if_exists: bool = False
) -> tuple[str, bool, str]:
    """解压单个 tar 到 dest_dir。返回 (tar_path.name, success, message)。"""
    if dry_run:
        return (tar_path.name, True, "would extract")

    top = None
    if skip_if_exists:
        top = _tar_first_top_dir(tar_path)
        if top:
            marker = dest_dir / top / EXTRACTED_MARKER
            if marker.exists():
                return (tar_path.name, True, "skipped")
            incomplete = dest_dir / top
            if incomplete.exists():
                try:
                    shutil.rmtree(incomplete)
                except OSError:
                    pass

    try:
        with tarfile.open(tar_path, "r") as tf:
            if top is None:
                first = next(iter(tf), None)
                top = first.name.split("/")[0] if first and first.name else None
            tf.extractall(dest_dir)
        if top:
            (dest_dir / top / EXTRACTED_MARKER).touch()
        return (tar_path.name, True, "ok")
    except Exception as e:
        return (tar_path.name, False, str(e))


def _progress_line(n_done: int, total: int, bytes_done: int, start_time: float) -> str:
    """生成一行进度：进度条 + 百分比 + 已解压 MB + 平均速度 MB/s。"""
    if total <= 0:
        return ""
    elapsed = max(time.time() - start_time, 1e-6)
    pct = 100.0 * n_done / total
    filled = int(PROGRESS_BAR_WIDTH * n_done / total)
    bar = "█" * filled + "░" * (PROGRESS_BAR_WIDTH - filled)
    mb_done = bytes_done / (1024 * 1024)
    speed_mbs = bytes_done / elapsed / (1024 * 1024)
    return f"  [{bar}] {n_done}/{total} ({pct:.1f}%) | {mb_done:.0f} MB 已解压 | 平均 {speed_mbs:.1f} MB/s"


def main():
    parser = argparse.ArgumentParser(description="解压 AgiBot-World observations 与 proprio_stats 的 tar")
    parser.add_argument("--root", type=Path, required=True, help="数据根目录，即 raw/main")
    parser.add_argument("--jobs", type=int, default=1, help="并行解压的线程数，默认 1")
    parser.add_argument("--dry-run", action="store_true", help="只列出要解压的 tar，不实际解压")
    parser.add_argument("--skip-proprio", action="store_true", help="跳过 proprio_stats.tar 解压")
    parser.add_argument("--skip-observations", action="store_true", help="跳过 observations 下各 task 的 tar 解压")
    parser.add_argument("--resume", action="store_true", help="断点续传：已解压过的 tar（目标目录已存在）则跳过")
    parser.add_argument("--progress-interval", type=float, default=PROGRESS_REFRESH_INTERVAL, help="进度条刷新间隔（秒），默认 1.0")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"不是目录: {root}")

    # 1) proprio_stats: 单个 proprio_stats.tar -> 解压到 proprio_stats/
    proprio_tar = root / "proprio_stats" / "proprio_stats.tar"
    tasks_done = 0
    tasks_fail = 0

    if not args.skip_proprio and proprio_tar.is_file():
        dest = root / "proprio_stats"
        dest.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            print(f"[dry-run] would extract {proprio_tar} -> {dest}")
        elif args.resume and (dest / EXTRACTED_MARKER).exists():
            print(f"Resume: {proprio_tar.name} already extracted (marker exists), skipping", flush=True)
        else:
            print(f"Extracting {proprio_tar.name} -> {dest} (streaming) ...", flush=True)
            try:
                use_cr = sys.stdout.isatty()
                t0 = time.time()
                n_members = 0
                with tarfile.open(proprio_tar, "r") as tf:
                    for member in tf:
                        tf.extract(member, dest, set_attrs=False)
                        n_members += 1
                        if n_members % 200 == 0:
                            elapsed = max(time.time() - t0, 1e-6)
                            rate = n_members / elapsed
                            msg = f"  proprio: {n_members} files | {rate:.0f} files/s | {elapsed:.0f}s"
                            if use_cr:
                                print(f"\r{msg}", end="", flush=True)
                            else:
                                print(msg, flush=True)
                elapsed = time.time() - t0
                if use_cr:
                    print(flush=True)
                (dest / EXTRACTED_MARKER).touch()
                tasks_done += 1
                print(f"  done: {proprio_tar.name} ({n_members} files, {elapsed:.0f}s)")
            except Exception as e:
                if use_cr:
                    print(flush=True)
                tasks_fail += 1
                print(f"  FAILED: {e}")
    elif not args.skip_proprio:
        print(f"Skip: not found {proprio_tar}")

    # 2) observations: 每个 task 目录下多个 *.tar -> 解压到 observations/{task_id}/
    obs_dir = root / "observations"
    if not args.skip_observations and obs_dir.is_dir():
        tar_list = []
        for task_dir in sorted(obs_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for f in task_dir.iterdir():
                if f.suffix == ".tar" and f.is_file():
                    try:
                        size = f.stat().st_size
                    except OSError:
                        size = 0
                    tar_list.append((f, task_dir, size))

        total = len(tar_list)
        total_bytes = sum(item[2] for item in tar_list)
        total_mb = total_bytes / (1024 * 1024)
        print(f"Found {total} observation tars under {obs_dir} ({total_mb:.0f} MB total)", flush=True)
        if args.dry_run:
            for tar_path, dest, _ in tar_list[:20]:
                print(f"  [dry-run] {tar_path.relative_to(root)} -> {dest.relative_to(root)}")
            if total > 20:
                print(f"  ... and {total - 20} more")
        else:
            if args.resume:
                print("Resume: skipping tars whose content dir already exists", flush=True)

            def do_one(item):
                tar_path, dest, _ = item
                return extract_one(tar_path, dest, dry_run=False, skip_if_exists=args.resume)

            start_time = time.time()
            state = {"tasks_done": 0, "tasks_fail": 0, "bytes_done": 0}
            lock = threading.Lock()
            use_cr = sys.stdout.isatty()
            interval = max(0.2, args.progress_interval)

            def progress_reporter():
                while True:
                    time.sleep(interval)
                    with lock:
                        n = state["tasks_done"] + state["tasks_fail"]
                        b = state["bytes_done"]
                    if n >= total:
                        break
                    line = _progress_line(n, total, b, start_time)
                    if line and use_cr:
                        print(f"\r{line}", end="", flush=True)

            reporter = threading.Thread(target=progress_reporter, daemon=True)
            reporter.start()

            try:
                if args.jobs <= 1:
                    for item in tar_list:
                        name, ok, msg = do_one(item)
                        with lock:
                            if ok:
                                state["tasks_done"] += 1
                                state["bytes_done"] += item[2]
                            else:
                                state["tasks_fail"] += 1
                                print(f"\nFAILED {name}: {msg}", flush=True)
                else:
                    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                        futures = {ex.submit(do_one, item): item for item in tar_list}
                        for fut in as_completed(futures):
                            item = futures[fut]
                            name, ok, msg = fut.result()
                            with lock:
                                if ok:
                                    state["tasks_done"] += 1
                                    state["bytes_done"] += item[2]
                                else:
                                    state["tasks_fail"] += 1
                                    print(f"\nFAILED {name}: {msg}", flush=True)
            finally:
                with lock:
                    n = state["tasks_done"] + state["tasks_fail"]
                    b = state["bytes_done"]
                line = _progress_line(n, total, b, start_time)
                if line:
                    if use_cr:
                        print(f"\r{line}", flush=True)
                    else:
                        print(line, flush=True)
                tasks_done += state["tasks_done"]
                tasks_fail += state["tasks_fail"]

    print(f"\nDone. OK: {tasks_done}, Failed: {tasks_fail}", flush=True)
    if tasks_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
