#!/usr/bin/env python3
"""Small contract check for the pin-aware rectified guidance model."""

from pathlib import Path
import sys

import torch

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

from training_example import DiTSmallFloorplanBackbone, build_initial_layout_batch, compute_batch_loss


def main():
    area = torch.tensor([[4.0, 9.0, 1.0]], dtype=torch.float32)
    constraints = torch.zeros((1, 3, 5), dtype=torch.float32)
    fp_sol = torch.tensor([[[2.0, 2.0, 0.0, 0.0], [3.0, 3.0, 3.0, 0.0], [1.0, 1.0, 0.0, 3.0]]])
    b2b = torch.tensor([[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5], [-1.0, -1.0, -1.0]]])
    p2b = torch.tensor([[[0.0, 1.0, 2.0], [1.0, 2.0, 1.0], [-1.0, -1.0, -1.0]]])
    pins = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
    block_mask = area > 0

    layout = build_initial_layout_batch(area, constraints, fp_sol)
    model = DiTSmallFloorplanBackbone(hidden_size=24, depth=1, num_heads=4)
    assert int(model.state_dict()["coordinate_contract_version"].item()) == 2
    out = model(layout, area.unsqueeze(-1), constraints, b2b, p2b, pins, torch.zeros(1, dtype=torch.long), block_mask)

    assert out.shape == layout.shape
    assert torch.isfinite(out).all()

    tree = torch.full((1, 2, 3), -1.0)
    metrics = torch.tensor([[20.0, 2.0, 2.0, 2.0, 2.0, 0.0, 10.0, 10.0]])
    loss, proxy_loss, xy_loss = compute_batch_loss(
        model,
        [area, b2b, p2b, pins, constraints, tree, fp_sol, metrics],
        torch.device("cpu"),
        max_blocks=3,
        noise_std=0.0,
        supervised_weight=1.0,
        return_components=True,
    )
    assert loss is not None
    assert torch.isfinite(loss).all()
    assert torch.isfinite(proxy_loss).all()
    assert torch.isfinite(xy_loss).all()


if __name__ == "__main__":
    main()
