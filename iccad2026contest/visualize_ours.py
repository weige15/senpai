#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_case(path, test_id):
    with open(path) as f:
        data = json.load(f)

    cases = data.get("test_results") or data.get("solutions") or []
    for case in cases:
        if int(case["test_id"]) == test_id:
            return data, case
    raise SystemExit(f"test_id {test_id} not found in {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize our ICCAD floorplan result JSON.")
    parser.add_argument("json_path", help="Evaluator result JSON or --save-solutions JSON")
    parser.add_argument("--test-id", "-t", type=int, default=0)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data, case = load_case(args.json_path, args.test_id)
    positions = case.get("positions") or []
    if not positions:
        raise SystemExit(f"test_id {args.test_id} has no positions")

    fig, ax = plt.subplots(figsize=(8, 8))
    for i, (x, y, w, h) in enumerate(positions):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="black", linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, str(i), ha="center", va="center", fontsize=6)

    name = data.get("submission_name") or data.get("submission") or Path(args.json_path).stem
    title = f"{name} case {args.test_id}"
    if "cost" in case:
        title += f", cost={case['cost']:.4g}, feasible={case.get('is_feasible')}"
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.3)

    output = args.output or f"case_{args.test_id}_ours.png"
    plt.savefig(output, dpi=180, bbox_inches="tight")
    print(f"saved {output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
