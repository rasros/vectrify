#!/usr/bin/env python3
"""
Analyze a speedscope JSON profile (from py-spy) for CPU hotspots.

Usage:
    uv tool run py-spy record -f speedscope -o profile.json \\
        --subprocesses -- uv run vectrify ...
    python scripts/analyze_profile.py profile.json
"""

import json
import sys
from collections import Counter
from pathlib import Path


def shorten_path(path: str) -> str:
    for prefix in [
        "/home/rasmus/Workspaces/image-to-svg/src/",
        "/home/rasmus/Workspaces/image-to-svg/",
    ]:
        if path.startswith(prefix):
            return path[len(prefix) :]
    if "site-packages/" in path:
        return "…/" + path.split("site-packages/")[-1]
    if "linuxbrew" in path or "opt/python" in path:
        parts = path.split("/")
        return "…/" + "/".join(parts[-3:])
    return path


def top_frames(
    profile: dict, shared_frames: list, n: int = 20
) -> tuple[list[tuple[int, int, str, str, int | str]], float]:
    samples = profile.get("samples", [])
    weights = profile.get("weights", [])
    total = sum(weights)
    counts: Counter[int] = Counter()
    for stack in samples:
        for fid in stack:
            counts[fid] += 1
    result = []
    for fid, cnt in counts.most_common(n):
        if fid < len(shared_frames):
            f = shared_frames[fid]
            result.append(
                (
                    cnt,
                    fid,
                    f.get("name", "?"),
                    shorten_path(f.get("file", "")),
                    f.get("line", ""),
                )
            )
    return result, total


def main(path: str) -> None:
    print(f"Loading {path} ...")
    with Path(path).open() as f:
        data = json.load(f)

    shared_frames = data.get("shared", {}).get("frames", [])
    profiles: list[dict] = data.get("profiles", [])

    print(f"Shared frames: {len(shared_frames)}")
    print(f"Total profiles: {len(profiles)}")
    print()

    # Split main thread (profile 0) from workers
    main_profile = None
    worker_profiles = []
    for p in profiles:
        if p.get("type") != "sampled":
            continue
        name = p.get("name", "")
        if main_profile is None and "MainThread" in name:
            main_profile = p
        elif "QueueFeederThread" not in name:
            worker_profiles.append(p)

    # --- Main thread ---
    if main_profile:
        frames, total = top_frames(main_profile, shared_frames)
        print(f"=== Main thread: {total:.1f}s ===")
        print(f"  {'Count':>7}  {'%':>5}  Frame")
        n_samples = max(1, len(main_profile["samples"]))
        for cnt, _, name, file_, line in frames:
            pct = 100 * cnt / n_samples
            print(f"  {cnt:7d}  {pct:4.1f}%  {name}  [{file_}:{line}]")
        print()

    # --- Workers aggregated ---
    if worker_profiles:
        combined_counts: Counter[int] = Counter()
        total_worker_weight = 0.0
        total_worker_samples = 0
        for p in worker_profiles:
            total_worker_weight += sum(p.get("weights", []))
            total_worker_samples += len(p.get("samples", []))
            for stack in p.get("samples", []):
                for fid in stack:
                    combined_counts[fid] += 1

        nw = len(worker_profiles)
        print(f"=== Workers ({nw} threads): {total_worker_weight:.1f}s ===")
        print(f"  {'Count':>7}  {'%':>5}  Frame")
        for fid, cnt in combined_counts.most_common(25):
            if fid < len(shared_frames):
                f = shared_frames[fid]
                name = f.get("name", "?")
                file_ = shorten_path(f.get("file", ""))
                line = f.get("line", "")
                pct = 100 * cnt / max(1, total_worker_samples)
                print(f"  {cnt:7d}  {pct:4.1f}%  {name}  [{file_}:{line}]")
        print()

    # --- Summary ---
    main_w = sum(main_profile.get("weights", [])) if main_profile else 0
    worker_w = sum(sum(p.get("weights", [])) for p in worker_profiles)
    total_w = main_w + worker_w
    print("=== CPU time summary ===")
    print(f"  Main thread : {main_w:7.1f}s  ({100 * main_w / max(1, total_w):.1f}%)")
    print(
        f"  Workers     : {worker_w:7.1f}s  ({100 * worker_w / max(1, total_w):.1f}%)"
    )
    print(f"  Total       : {total_w:7.1f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <profile.json>")
        sys.exit(1)
    main(sys.argv[1])
