#!/usr/bin/env python
"""
Build a mapping from LeRobot episode_index -> RLDS episode metadata (recording_folderpath, file_path).

Uses the exact same ReadConfig as the original openx_rlds.py conversion so iteration order matches.
Skips image decoding for speed. Outputs info/droid_mapping.jsonl.

Usage:
    CUDA_VISIBLE_DEVICES="" python build_droid_mapping.py \
        --raw-dir /mnt/data/data/open/RT-X/tensorflow_datasets/droid/1.0.0 \
        --lerobot-dir /tmp/droid_lerobot/droid_1.0.0_lerobot \
        --output /home/fox/vla_linear/any4lerobot/info/droid_mapping.jsonl
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds


CHECKPOINT_INTERVAL = 1000


def _iter_episodes_safe(dataset):
    it = dataset.as_numpy_iterator()
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except tf.errors.DataLossError as e:
            print(f"WARNING: skipping corrupted record: {e}", flush=True)
            continue


def _load_episodes_jsonl(lerobot_dir):
    """Load LeRobot episodes.jsonl into {episode_index: length} dict."""
    path = Path(lerobot_dir) / "meta" / "episodes.jsonl"
    ep_lengths = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            ep_lengths[ep["episode_index"]] = ep["length"]
    return ep_lengths


def _load_checkpoint(output_path):
    ckpt_path = Path(output_path).with_suffix(".checkpoint.json")
    if not ckpt_path.exists():
        return 0
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    return ckpt.get("completed", 0)


def _save_checkpoint(output_path, completed):
    ckpt_path = Path(output_path).with_suffix(".checkpoint.json")
    with open(ckpt_path, "w") as f:
        json.dump({"completed": completed}, f)


def build_mapping(raw_dir: Path, lerobot_dir: Path, output: Path):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(raw_dir)
    last_part = raw_dir.name
    import re
    if re.match(r"^\d+\.\d+\.\d+$", last_part):
        version = last_part
        dataset_name = raw_dir.parent.name
        data_dir = raw_dir.parent.parent
    else:
        version = ""
        dataset_name = last_part
        data_dir = raw_dir.parent

    print(f"Loading TFDS builder: {dataset_name} v{version} from {data_dir}", flush=True)
    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)

    try:
        total_episodes = builder.info.splits["train"].num_examples
    except Exception:
        total_episodes = None
    print(f"Total episodes: {total_episodes}", flush=True)

    ep_lengths = _load_episodes_jsonl(lerobot_dir)
    print(f"Loaded {len(ep_lengths)} episodes from LeRobot episodes.jsonl", flush=True)

    n_skip = _load_checkpoint(output)
    if n_skip > 0:
        print(f"Resuming from episode {n_skip}", flush=True)

    read_config = tfds.ReadConfig(
        interleave_cycle_length=64,
        interleave_block_length=1,
        num_parallel_calls_for_interleave_files=tf.data.AUTOTUNE,
    )

    # Use SkipDecoding to avoid decoding images (huge speedup)
    raw_dataset = builder.as_dataset(
        split="train",
        read_config=read_config,
        decoders={"steps": tfds.decode.SkipDecoding()},
    )

    # Map to extract only episode_metadata (drop steps to avoid any step-level processing)
    def extract_meta(episode):
        return episode["episode_metadata"]

    raw_dataset = raw_dataset.map(extract_meta, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    if n_skip > 0:
        raw_dataset = raw_dataset.skip(n_skip)

    t0 = time.time()
    n_mismatch = 0
    n_no_lerobot = 0

    mode = "a" if n_skip > 0 else "w"
    with open(output, mode) as out_f:
        for i, meta in enumerate(_iter_episodes_safe(raw_dataset)):
            ep_idx = n_skip + i

            recording_fp = meta.get("recording_folderpath", b"")
            file_path = meta.get("file_path", b"")

            if isinstance(recording_fp, bytes):
                recording_fp = recording_fp.decode("utf-8", errors="replace")
            if isinstance(file_path, bytes):
                file_path = file_path.decode("utf-8", errors="replace")

                    # Use lerobot_length from episodes.jsonl directly (no need to count steps)
            lerobot_len = ep_lengths.get(ep_idx)
            n_steps = lerobot_len
            mismatch = False
            if lerobot_len is None:
                n_no_lerobot += 1

            entry = {
                "lerobot_idx": ep_idx,
                "recording_folderpath": recording_fp,
                "file_path": file_path,
                "n_steps": n_steps,
                "lerobot_length": lerobot_len,
                "mismatch": mismatch,
            }
            out_f.write(json.dumps(entry) + "\n")

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                out_f.flush()
                _save_checkpoint(output, ep_idx + 1)
                elapsed = time.time() - t0
                eps_per_sec = (i + 1) / elapsed
                total_str = str(total_episodes) if total_episodes else "?"
                print(
                    f"Episode {ep_idx}/{total_str} | "
                    f"{eps_per_sec:.1f} ep/s | "
                    f"mismatches={n_mismatch} | "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

    elapsed = time.time() - t0
    total_processed = n_skip + i + 1
    print(f"\nDone! {total_processed} episodes processed in {elapsed:.0f}s", flush=True)
    print(f"Mismatches: {n_mismatch}", flush=True)
    print(f"Episodes not in LeRobot: {n_no_lerobot}", flush=True)
    print(f"Output: {output}", flush=True)

    # Clean up checkpoint
    ckpt_path = output.with_suffix(".checkpoint.json")
    if ckpt_path.exists():
        ckpt_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--lerobot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("info/droid_mapping.jsonl"))
    args = parser.parse_args()
    build_mapping(args.raw_dir, args.lerobot_dir, args.output)


if __name__ == "__main__":
    main()
