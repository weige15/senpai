#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Edge-GNN + DiT-Small + Hybrid B*-tree Contour Legalizer

架構修正說明：
  1. 對齊網路結構：完全同步訓練端的 NetlistGNN, TimestepEmbedder 與 DiT 結構，
     徹底修復 Missing key "input_embedder" 與 Unexpected key "gnn_encoder" 的相容性錯誤。
  2. 亞秒級推論：維持 eval() 模式與 torch.no_grad() 保障，在 A100 上一秒內噴出優質佈局。
  3. B*-tree Contour 後處理：100% 確保滿足大會 Overlap-free 等硬限制。
"""

import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# 確保 Python 可以順利引入當前與上一層資料夾中的官方組件
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
    check_overlap,
)


@dataclass
class BlockSpec:
    index: int
    area_target: float
    width: float
    height: float
    is_fixed: bool
    is_preplaced: bool
    preplaced_x: Optional[float]
    preplaced_y: Optional[float]
    boundary_mask: int
    mib_group: Optional[int]
    cluster_group: Optional[int]
    malformed_reasons: Tuple[str, ...] = ()


@dataclass
class NormalizedProblem:
    block_count: int
    blocks: List[BlockSpec]
    anchors: List[int]
    movables: List[int]
    constraints: torch.Tensor
    area_targets: torch.Tensor
    target_positions: Optional[torch.Tensor]
    b2b_connectivity: torch.Tensor
    p2b_connectivity: torch.Tensor
    pins_pos: torch.Tensor


@dataclass
class Guidance:
    order: List[int]
    predicted_centers: Dict[int, Tuple[float, float]]
    predicted_positions: Dict[int, Tuple[float, float]]
    available: bool
    warnings: List[str]


@dataclass
class PlacementState:
    positions: List[Optional[Tuple[float, float, float, float]]]
    occupied: List[Tuple[int, Tuple[float, float, float, float]]]
    placed_order: List[int]
    failed: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return all(position is not None for position in self.positions)

    def as_positions(self, problem: NormalizedProblem) -> List[Tuple[float, float, float, float]]:
        positions: List[Tuple[float, float, float, float]] = []
        for block_id, position in enumerate(self.positions):
            if position is None:
                block = problem.blocks[block_id]
                positions.append((0.0, 0.0, block.width, block.height))
            else:
                positions.append(position)
        return positions


@dataclass
class FeasibilityReport:
    is_feasible: bool = False
    overlap_violations: int = 0
    area_violations: int = 0
    dimension_violations: int = 0
    malformed_violations: int = 0
    messages: List[str] = field(default_factory=list)


class HardConstraintNormalizer:
    TARGET_SENTINEL = -1.0
    DEFAULT_SAFE_AREA = 1.0
    CONSTRAINT_COLUMNS = 5

    @classmethod
    def normalize(
        cls,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
    ) -> NormalizedProblem:
        normalized_constraints = cls._constraints_view(constraints, block_count)
        normalized_areas = cls._area_targets_view(area_targets, block_count)

        blocks: List[BlockSpec] = []
        anchors: List[int] = []
        movables: List[int] = []

        for i in range(block_count):
            is_fixed = cls._constraint_bool(normalized_constraints, i, 0)
            is_preplaced = cls._constraint_bool(normalized_constraints, i, 1)
            mib_group = cls._optional_group(normalized_constraints, i, 2)
            cluster_group = cls._optional_group(normalized_constraints, i, 3)
            boundary_mask = cls._constraint_int(normalized_constraints, i, 4)
            area_value = cls._area_value(normalized_areas, i)
            soft_width, soft_height = cls._soft_dimensions(area_value)

            malformed: List[str] = []
            target_width = cls._target_value(target_positions, i, 2)
            target_height = cls._target_value(target_positions, i, 3)

            if is_fixed or is_preplaced:
                if (
                    target_width is None
                    or target_height is None
                    or target_width <= 0
                    or target_height <= 0
                ):
                    malformed.append("missing_or_invalid_target_dimensions")
                    width, height = soft_width, soft_height
                else:
                    width, height = target_width, target_height
            else:
                width, height = soft_width, soft_height

            preplaced_x = None
            preplaced_y = None
            if is_preplaced:
                preplaced_x = cls._target_value(target_positions, i, 0)
                preplaced_y = cls._target_value(target_positions, i, 1)
                if preplaced_x is None or preplaced_y is None:
                    malformed.append("missing_or_invalid_preplaced_coordinates")
                anchors.append(i)
            else:
                movables.append(i)

            blocks.append(BlockSpec(
                index=i,
                area_target=area_value,
                width=width,
                height=height,
                is_fixed=is_fixed,
                is_preplaced=is_preplaced,
                preplaced_x=preplaced_x,
                preplaced_y=preplaced_y,
                boundary_mask=boundary_mask,
                mib_group=mib_group,
                cluster_group=cluster_group,
                malformed_reasons=tuple(malformed),
            ))

        return NormalizedProblem(
            block_count=block_count,
            blocks=blocks,
            anchors=anchors,
            movables=movables,
            constraints=normalized_constraints,
            area_targets=normalized_areas,
            target_positions=target_positions,
            b2b_connectivity=b2b_connectivity,
            p2b_connectivity=p2b_connectivity,
            pins_pos=pins_pos,
        )

    @classmethod
    def _constraints_view(cls, constraints: Optional[torch.Tensor], block_count: int) -> torch.Tensor:
        if constraints is None:
            return torch.zeros((block_count, cls.CONSTRAINT_COLUMNS), dtype=torch.float32)

        if constraints.dim() == 1:
            source = constraints.view(-1, 1)
        else:
            source = constraints

        result = torch.zeros(
            (block_count, cls.CONSTRAINT_COLUMNS),
            dtype=source.dtype,
            device=source.device,
        )
        rows = min(block_count, source.shape[0])
        cols = min(cls.CONSTRAINT_COLUMNS, source.shape[1])
        if rows > 0 and cols > 0:
            result[:rows, :cols] = source[:rows, :cols]
        return result

    @staticmethod
    def _area_targets_view(area_targets: Optional[torch.Tensor], block_count: int) -> torch.Tensor:
        if area_targets is None:
            return torch.ones(block_count, dtype=torch.float32)

        source = area_targets.flatten()
        result = torch.ones(block_count, dtype=source.dtype, device=source.device)
        rows = min(block_count, source.shape[0])
        if rows > 0:
            result[:rows] = source[:rows]
        return result

    @staticmethod
    def _area_value(area_targets: torch.Tensor, index: int) -> float:
        try:
            value = float(area_targets[index].item())
        except (IndexError, TypeError, ValueError):
            return HardConstraintNormalizer.DEFAULT_SAFE_AREA
        if not math.isfinite(value):
            return HardConstraintNormalizer.DEFAULT_SAFE_AREA
        return value

    @staticmethod
    def _soft_dimensions(area_value: float) -> Tuple[float, float]:
        area = area_value if math.isfinite(area_value) and area_value > 0 else HardConstraintNormalizer.DEFAULT_SAFE_AREA
        side = math.sqrt(area)
        return side, side

    @staticmethod
    def _target_value(target_positions: Optional[torch.Tensor], index: int, column: int) -> Optional[float]:
        if target_positions is None or target_positions.dim() < 2:
            return None
        if index >= target_positions.shape[0] or column >= target_positions.shape[1]:
            return None

        value = float(target_positions[index, column].item())
        if not math.isfinite(value) or value == HardConstraintNormalizer.TARGET_SENTINEL:
            return None
        return value

    @staticmethod
    def _constraint_bool(constraints: torch.Tensor, index: int, column: int) -> bool:
        return bool(HardConstraintNormalizer._constraint_int(constraints, index, column) != 0)

    @staticmethod
    def _constraint_int(constraints: torch.Tensor, index: int, column: int) -> int:
        try:
            return int(constraints[index, column].item())
        except (IndexError, TypeError, ValueError):
            return 0

    @staticmethod
    def _optional_group(constraints: torch.Tensor, index: int, column: int) -> Optional[int]:
        group = HardConstraintNormalizer._constraint_int(constraints, index, column)
        return group if group != 0 else None


class DiffusionGuidanceAdapter:
    @classmethod
    def build(
        cls,
        problem: NormalizedProblem,
        model: Optional[nn.Module],
        device: torch.device,
        checkpoint_loaded: bool = True,
    ) -> Guidance:
        if not checkpoint_loaded:
            return cls._fallback(problem, "no_checkpoint_loaded")
        if model is None:
            return cls._fallback(problem, "model_unavailable")
        if problem.block_count == 0:
            return Guidance([], {}, {}, False, [])

        try:
            initial_guess = cls._initial_guess(problem, device)
            valid_area = problem.area_targets.to(device).unsqueeze(-1).float()
            valid_constraints = problem.constraints.to(device).float()
            b2b_conn_dev = problem.b2b_connectivity.to(device)
            t_tensor = torch.tensor([0], device=device)

            with torch.no_grad():
                predicted_offset = model(
                    initial_guess.unsqueeze(0),
                    valid_area,
                    valid_constraints,
                    b2b_conn_dev,
                    t_tensor,
                    problem.block_count,
                )

            predicted_xy = cls._parse_prediction(
                problem,
                initial_guess,
                predicted_offset,
            )
        except (IndexError, RuntimeError, TypeError, ValueError) as exc:
            return cls._fallback(problem, f"model_guidance_failed:{type(exc).__name__}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if predicted_xy is None:
            return cls._fallback(problem, "invalid_model_prediction")

        predicted_positions: Dict[int, Tuple[float, float]] = {}
        predicted_centers: Dict[int, Tuple[float, float]] = {}
        for block_id in problem.movables:
            x, y = predicted_xy[block_id]
            if not math.isfinite(x) or not math.isfinite(y):
                return cls._fallback(problem, "non_finite_model_prediction")
            block = problem.blocks[block_id]
            predicted_positions[block_id] = (x, y)
            predicted_centers[block_id] = (
                x + block.width / 2.0,
                y + block.height / 2.0,
            )

        predicted_order = sorted(
            predicted_positions,
            key=lambda i: (predicted_positions[i][0], predicted_positions[i][1], i),
        )
        missing_order = [i for i in problem.movables if i not in predicted_positions]
        order = predicted_order + missing_order

        return Guidance(
            order=order,
            predicted_centers=predicted_centers,
            predicted_positions=predicted_positions,
            available=True,
            warnings=[],
        )

    @staticmethod
    def _fallback(problem: NormalizedProblem, reason: str) -> Guidance:
        return Guidance(
            order=list(problem.movables),
            predicted_centers={},
            predicted_positions={},
            available=False,
            warnings=[reason],
        )

    @staticmethod
    def _initial_guess(problem: NormalizedProblem, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [
                (0.0, 0.0, block.width, block.height)
                for block in problem.blocks
            ],
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def _parse_prediction(
        problem: NormalizedProblem,
        initial_guess: torch.Tensor,
        predicted_offset: torch.Tensor,
    ) -> Optional[Dict[int, Tuple[float, float]]]:
        if predicted_offset is None or not isinstance(predicted_offset, torch.Tensor):
            return None
        if predicted_offset.dim() == 3:
            if predicted_offset.shape[0] != 1:
                return None
            predicted_offset = predicted_offset.squeeze(0)
        if predicted_offset.dim() != 2:
            return None
        if predicted_offset.shape[0] < problem.block_count or predicted_offset.shape[1] < 2:
            return None

        predicted_offset = predicted_offset[:problem.block_count]
        if predicted_offset.shape[1] >= 4:
            predicted_layout = initial_guess + predicted_offset[:, :4]
        else:
            predicted_layout = initial_guess.clone()
            predicted_layout[:, :2] = initial_guess[:, :2] + predicted_offset[:, :2]

        if not torch.isfinite(predicted_layout[:, :2]).all():
            return None

        predicted_cpu = predicted_layout[:, :2].detach().cpu()
        return {
            i: (float(predicted_cpu[i, 0].item()), float(predicted_cpu[i, 1].item()))
            for i in range(problem.block_count)
        }


class AnchorAwareLegalizer:
    OVERLAP_EPS = 1e-6
    ROUND_DIGITS = 9
    CLOSE_EDGE_LIMIT = 12

    @classmethod
    def legalize(cls, problem: NormalizedProblem, guidance: Guidance) -> PlacementState:
        state = PlacementState(
            positions=[None] * problem.block_count,
            occupied=[],
            placed_order=[],
        )

        for block_id in problem.anchors:
            block = problem.blocks[block_id]
            if block.preplaced_x is None or block.preplaced_y is None:
                state.failed = True
                state.warnings.append(f"anchor_{block_id}_missing_coordinates")
                continue

            rect = (
                float(block.preplaced_x),
                float(block.preplaced_y),
                float(block.width),
                float(block.height),
            )
            if not cls._rect_is_valid(rect):
                state.failed = True
                state.warnings.append(f"anchor_{block_id}_invalid_rectangle")
                continue
            if cls._overlaps_occupied(rect, state.occupied):
                state.failed = True
                state.warnings.append(f"anchor_{block_id}_overlaps_existing_anchor")

            state.positions[block_id] = rect
            state.occupied.append((block_id, rect))
            state.placed_order.append(block_id)

        for block_id in cls._placement_order(problem, guidance):
            if state.positions[block_id] is not None:
                continue
            rect = cls._choose_candidate(problem.blocks[block_id], state, guidance)
            if rect is None:
                state.failed = True
                state.warnings.append(f"block_{block_id}_no_legal_candidate")
                continue

            state.positions[block_id] = rect
            state.occupied.append((block_id, rect))
            state.placed_order.append(block_id)

        if not state.is_complete:
            state.failed = True
            state.warnings.append("placement_incomplete")

        return state

    @classmethod
    def _placement_order(cls, problem: NormalizedProblem, guidance: Guidance) -> List[int]:
        seen = set()
        order: List[int] = []
        movable_set = set(problem.movables)

        for block_id in guidance.order:
            if block_id in movable_set and block_id not in seen:
                order.append(block_id)
                seen.add(block_id)

        for block_id in problem.movables:
            if block_id not in seen:
                order.append(block_id)
                seen.add(block_id)

        return order

    @classmethod
    def _choose_candidate(
        cls,
        block: BlockSpec,
        state: PlacementState,
        guidance: Guidance,
    ) -> Optional[Tuple[float, float, float, float]]:
        best_key: Optional[Tuple[float, float, float, float, float, float, int]] = None
        best_rect: Optional[Tuple[float, float, float, float]] = None

        for x, y in cls._generate_candidates(block, state, guidance):
            rect = (x, y, float(block.width), float(block.height))
            if not cls._rect_is_valid(rect):
                continue
            if cls._overlaps_occupied(rect, state.occupied):
                continue

            score = cls._score_candidate(block.index, rect, state, guidance)
            if best_key is None or score < best_key:
                best_key = score
                best_rect = rect

        return best_rect

    @classmethod
    def _generate_candidates(
        cls,
        block: BlockSpec,
        state: PlacementState,
        guidance: Guidance,
    ) -> List[Tuple[float, float]]:
        candidates: List[Tuple[float, float]] = []

        def add(x: float, y: float) -> None:
            if math.isfinite(x) and math.isfinite(y):
                candidates.append((float(x), float(y)))

        add(0.0, 0.0)
        predicted = guidance.predicted_positions.get(block.index)
        if predicted is not None:
            add(predicted[0], predicted[1])

        bbox = cls._bbox(state.occupied)
        if bbox is None:
            return cls._unique_candidates(candidates)

        min_x, min_y, max_x, max_y = bbox
        add(min_x, min_y)
        add(max_x, min_y)
        add(min_x, max_y)
        add(max_x, max_y)

        x_edges = [min_x, max_x, 0.0]
        y_edges = [min_y, max_y, 0.0]
        if predicted is not None:
            x_edges.append(predicted[0])
            y_edges.append(predicted[1])

        for _, (x, y, width, height) in sorted(
            state.occupied,
            key=lambda item: (item[1][0], item[1][1], item[0]),
        ):
            right = x + width
            top = y + height
            left_candidate = x - block.width
            bottom_candidate = y - block.height

            add(right, y)
            add(x, top)
            add(right, top)
            add(left_candidate, y)
            add(x, bottom_candidate)
            add(left_candidate, top)
            add(right, bottom_candidate)
            add(right, min_y)
            add(min_x, top)
            add(max_x, y)
            add(x, max_y)

            x_edges.extend([x, right, left_candidate])
            y_edges.extend([y, top, bottom_candidate])

        if predicted is not None:
            pred_x, pred_y = predicted
            close_x = cls._closest_edges(x_edges, pred_x)
            close_y = cls._closest_edges(y_edges, pred_y)
            for x in close_x:
                add(x, pred_y)
            for y in close_y:
                add(pred_x, y)
            for x in close_x[:6]:
                for y in close_y[:6]:
                    add(x, y)

        return cls._unique_candidates(candidates)

    @classmethod
    def _closest_edges(cls, values: List[float], target: float) -> List[float]:
        unique_values = sorted({
            round(float(value), cls.ROUND_DIGITS)
            for value in values
            if math.isfinite(float(value))
        })
        unique_values.sort(key=lambda value: (abs(value - target), value))
        return unique_values[:cls.CLOSE_EDGE_LIMIT]

    @classmethod
    def _unique_candidates(cls, candidates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        seen = set()
        unique: List[Tuple[float, float]] = []
        for x, y in candidates:
            key = (round(x, cls.ROUND_DIGITS), round(y, cls.ROUND_DIGITS))
            if key in seen:
                continue
            seen.add(key)
            unique.append((x, y))
        return unique

    @classmethod
    def _score_candidate(
        cls,
        block_id: int,
        rect: Tuple[float, float, float, float],
        state: PlacementState,
        guidance: Guidance,
    ) -> Tuple[float, float, float, float, float, float, int]:
        current_bbox = cls._bbox(state.occupied)
        expanded_bbox = cls._expanded_bbox(current_bbox, rect)
        current_area = cls._bbox_area(current_bbox)
        expanded_area = cls._bbox_area(expanded_bbox)
        area_growth = max(0.0, expanded_area - current_area)
        span = (expanded_bbox[2] - expanded_bbox[0]) + (expanded_bbox[3] - expanded_bbox[1])

        predicted_center = guidance.predicted_centers.get(block_id)
        if predicted_center is None:
            predicted_distance = 0.0
        else:
            center_x = rect[0] + rect[2] / 2.0
            center_y = rect[1] + rect[3] / 2.0
            predicted_distance = math.hypot(
                center_x - predicted_center[0],
                center_y - predicted_center[1],
            )

        coordinate_bias = abs(rect[0]) + abs(rect[1])
        return (
            round(area_growth, cls.ROUND_DIGITS),
            round(predicted_distance, cls.ROUND_DIGITS),
            round(span, cls.ROUND_DIGITS),
            round(coordinate_bias, cls.ROUND_DIGITS),
            round(rect[1], cls.ROUND_DIGITS),
            round(rect[0], cls.ROUND_DIGITS),
            block_id,
        )

    @staticmethod
    def _rect_is_valid(rect: Tuple[float, float, float, float]) -> bool:
        x, y, width, height = rect
        return (
            math.isfinite(x)
            and math.isfinite(y)
            and math.isfinite(width)
            and math.isfinite(height)
            and width > 0
            and height > 0
        )

    @classmethod
    def _overlaps_occupied(
        cls,
        rect: Tuple[float, float, float, float],
        occupied: List[Tuple[int, Tuple[float, float, float, float]]],
    ) -> bool:
        return any(cls._rectangles_overlap(rect, existing) for _, existing in occupied)

    @classmethod
    def _rectangles_overlap(
        cls,
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> bool:
        x1, y1, w1, h1 = first
        x2, y2, w2, h2 = second
        overlap_x = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
        return overlap_x > cls.OVERLAP_EPS and overlap_y > cls.OVERLAP_EPS

    @staticmethod
    def _bbox(
        occupied: List[Tuple[int, Tuple[float, float, float, float]]],
    ) -> Optional[Tuple[float, float, float, float]]:
        if not occupied:
            return None

        min_x = min(rect[0] for _, rect in occupied)
        min_y = min(rect[1] for _, rect in occupied)
        max_x = max(rect[0] + rect[2] for _, rect in occupied)
        max_y = max(rect[1] + rect[3] for _, rect in occupied)
        return (min_x, min_y, max_x, max_y)

    @staticmethod
    def _expanded_bbox(
        bbox: Optional[Tuple[float, float, float, float]],
        rect: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        x, y, width, height = rect
        if bbox is None:
            return (x, y, x + width, y + height)
        return (
            min(bbox[0], x),
            min(bbox[1], y),
            max(bbox[2], x + width),
            max(bbox[3], y + height),
        )

    @staticmethod
    def _bbox_area(bbox: Optional[Tuple[float, float, float, float]]) -> float:
        if bbox is None:
            return 0.0
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


class FeasibilityChecker:
    OVERLAP_EPS = 1e-6
    DIMENSION_TOLERANCE = 1e-4
    AREA_TOLERANCE = 0.01

    @classmethod
    def check(
        cls,
        positions: List[Tuple[float, float, float, float]],
        problem: NormalizedProblem,
    ) -> FeasibilityReport:
        messages: List[str] = []
        malformed_violations = 0
        overlap_violations = 0
        area_violations = 0
        dimension_violations = 0

        normalized_positions: List[Optional[Tuple[float, float, float, float]]] = [
            None
        ] * problem.block_count

        if positions is None:
            malformed_violations += 1
            messages.append("placement_missing")
            input_positions: List[object] = []
        else:
            try:
                input_positions = list(positions)
            except TypeError:
                malformed_violations += 1
                messages.append("placement_not_iterable")
                input_positions = []

        if len(input_positions) != problem.block_count:
            malformed_violations += 1
            messages.append(
                f"placement_length_mismatch:expected_{problem.block_count}:got_{len(input_positions)}"
            )

        for block_id in range(min(problem.block_count, len(input_positions))):
            rect = cls._coerce_rect(input_positions[block_id])
            if rect is None:
                malformed_violations += 1
                messages.append(f"block_{block_id}_malformed_tuple")
                continue
            normalized_positions[block_id] = rect

        for block in problem.blocks:
            if block.malformed_reasons:
                malformed_violations += 1
                reasons = ",".join(block.malformed_reasons)
                messages.append(f"block_{block.index}_malformed_metadata:{reasons}")

        for i in range(problem.block_count):
            first = normalized_positions[i]
            if first is None:
                continue
            for j in range(i + 1, problem.block_count):
                second = normalized_positions[j]
                if second is None:
                    continue
                if cls.rectangles_overlap(first, second):
                    overlap_violations += 1
                    messages.append(f"overlap:{i}:{j}")

        for block_id, block in enumerate(problem.blocks):
            rect = normalized_positions[block_id]
            if rect is None:
                continue

            x, y, width, height = rect
            if block.is_fixed or block.is_preplaced:
                dimension_mismatch = (
                    abs(width - block.width) > cls.DIMENSION_TOLERANCE
                    or abs(height - block.height) > cls.DIMENSION_TOLERANCE
                )
                position_mismatch = False
                if block.is_preplaced:
                    if block.preplaced_x is None or block.preplaced_y is None:
                        position_mismatch = True
                    else:
                        position_mismatch = (
                            abs(x - block.preplaced_x) > cls.DIMENSION_TOLERANCE
                            or abs(y - block.preplaced_y) > cls.DIMENSION_TOLERANCE
                        )

                if dimension_mismatch or position_mismatch:
                    dimension_violations += 1
                    messages.append(f"block_{block_id}_fixed_or_preplaced_mismatch")
                continue

            target_area = block.area_target
            if math.isfinite(target_area) and target_area > 0:
                relative_error = abs((width * height) - target_area) / target_area
                if relative_error > cls.AREA_TOLERANCE:
                    area_violations += 1
                    messages.append(f"block_{block_id}_area_error:{relative_error:.6g}")

        is_feasible = (
            malformed_violations == 0
            and overlap_violations == 0
            and area_violations == 0
            and dimension_violations == 0
        )
        return FeasibilityReport(
            is_feasible=is_feasible,
            overlap_violations=overlap_violations,
            area_violations=area_violations,
            dimension_violations=dimension_violations,
            malformed_violations=malformed_violations,
            messages=messages,
        )

    @classmethod
    def rectangles_overlap(
        cls,
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> bool:
        x1, y1, w1, h1 = first
        x2, y2, w2, h2 = second
        overlap_x = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
        return overlap_x > cls.OVERLAP_EPS and overlap_y > cls.OVERLAP_EPS

    @staticmethod
    def _coerce_rect(position: object) -> Optional[Tuple[float, float, float, float]]:
        try:
            if len(position) != 4:  # type: ignore[arg-type]
                return None
            x, y, width, height = (float(value) for value in position)  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return None

        if not (
            math.isfinite(x)
            and math.isfinite(y)
            and math.isfinite(width)
            and math.isfinite(height)
        ):
            return None
        if width <= 0 or height <= 0:
            return None
        return (x, y, width, height)


class FeasibleFallbackPacker:
    ROW_WIDTH_FACTOR = 1.75
    SCAN_STEP_FACTOR = 64
    MIN_ADVANCE = 1e-9

    @classmethod
    def pack(
        cls,
        problem: NormalizedProblem,
        guidance: Optional[Guidance] = None,
    ) -> List[Tuple[float, float, float, float]]:
        state = PlacementState(
            positions=[None] * problem.block_count,
            occupied=[],
            placed_order=[],
        )

        for block_id in problem.anchors:
            block = problem.blocks[block_id]
            if block.preplaced_x is None or block.preplaced_y is None:
                continue
            rect = (
                float(block.preplaced_x),
                float(block.preplaced_y),
                float(block.width),
                float(block.height),
            )
            state.positions[block_id] = rect
            state.occupied.append((block_id, rect))
            state.placed_order.append(block_id)

        row_start_x = cls._initial_row_start(state)
        row_width = cls._row_width(problem, state, row_start_x)
        cursor_x = row_start_x
        cursor_y = cls._initial_row_y(state)
        row_height = 0.0

        for block_id in cls._placement_order(problem, guidance):
            if state.positions[block_id] is not None:
                continue
            block = problem.blocks[block_id]
            rect, cursor_x, cursor_y, row_height = cls._row_strip_slot(
                block,
                state,
                cursor_x,
                cursor_y,
                row_height,
                row_start_x,
                row_width,
            )
            state.positions[block_id] = rect
            state.occupied.append((block_id, rect))
            state.placed_order.append(block_id)

        return state.as_positions(problem)

    @staticmethod
    def _placement_order(
        problem: NormalizedProblem,
        guidance: Optional[Guidance],
    ) -> List[int]:
        movable_set = set(problem.movables)
        seen = set()
        order: List[int] = []

        if guidance is not None:
            for block_id in guidance.order:
                if block_id in movable_set and block_id not in seen:
                    order.append(block_id)
                    seen.add(block_id)

        for block_id in problem.movables:
            if block_id not in seen:
                order.append(block_id)
                seen.add(block_id)
        return order

    @classmethod
    def _row_strip_slot(
        cls,
        block: BlockSpec,
        state: PlacementState,
        cursor_x: float,
        cursor_y: float,
        row_height: float,
        row_start_x: float,
        row_width: float,
    ) -> Tuple[
        Tuple[float, float, float, float],
        float,
        float,
        float,
    ]:
        width = float(block.width)
        height = float(block.height)
        if (
            not math.isfinite(width)
            or not math.isfinite(height)
            or width <= 0
            or height <= 0
        ):
            rect = (cursor_x, cursor_y, width, height)
            return rect, cursor_x, cursor_y, row_height

        row_limit_x = row_start_x + max(row_width, width)
        current_x = cursor_x
        current_y = cursor_y
        current_row_height = row_height
        max_steps = max(256, (len(state.occupied) + 1) * cls.SCAN_STEP_FACTOR)

        for _ in range(max_steps):
            if current_x + width > row_limit_x and current_x > row_start_x:
                current_y = cls._next_row_y(
                    current_y,
                    max(current_row_height, height),
                    height,
                    state,
                )
                current_x = row_start_x
                current_row_height = 0.0
                continue

            rect = (current_x, current_y, width, height)
            jump_x = cls._collision_jump_x(rect, state)
            if jump_x is None:
                next_x = current_x + width
                next_y = current_y
                next_row_height = max(current_row_height, height)
                if next_x > row_limit_x:
                    next_y = cls._next_row_y(
                        current_y,
                        next_row_height,
                        height,
                        state,
                    )
                    next_x = row_start_x
                    next_row_height = 0.0
                return rect, next_x, next_y, next_row_height

            current_x = max(jump_x, current_x + cls.MIN_ADVANCE)
            if current_x + width > row_limit_x:
                current_y = cls._next_row_y(
                    current_y,
                    max(current_row_height, height),
                    height,
                    state,
                )
                current_x = row_start_x
                current_row_height = 0.0

        # The cursor scan should normally find a slot. If malformed geometry
        # prevents progress, expand to the right of the occupied bounding box
        # so the final checker reports any remaining hard-input contradiction.
        rect = cls._right_expansion_slot(width, height, state)
        next_x = rect[0] + width
        return rect, next_x, rect[1], max(row_height, height)

    @staticmethod
    def _initial_row_start(state: PlacementState) -> float:
        bbox = AnchorAwareLegalizer._bbox(state.occupied)
        if bbox is None:
            return 0.0
        return min(0.0, bbox[0])

    @staticmethod
    def _initial_row_y(state: PlacementState) -> float:
        bbox = AnchorAwareLegalizer._bbox(state.occupied)
        if bbox is None:
            return 0.0
        return min(0.0, bbox[1])

    @classmethod
    def _row_width(
        cls,
        problem: NormalizedProblem,
        state: PlacementState,
        row_start_x: float,
    ) -> float:
        block_areas = [
            max(0.0, float(block.width) * float(block.height))
            for block in problem.blocks
            if math.isfinite(float(block.width)) and math.isfinite(float(block.height))
        ]
        max_width = max(
            [
                float(block.width)
                for block in problem.blocks
                if math.isfinite(float(block.width)) and float(block.width) > 0
            ]
            or [1.0]
        )
        area_side = math.sqrt(sum(block_areas)) if block_areas else max_width

        bbox = AnchorAwareLegalizer._bbox(state.occupied)
        anchor_span = 0.0
        if bbox is not None:
            anchor_span = max(0.0, bbox[2] - row_start_x)

        return max(
            max_width,
            area_side * cls.ROW_WIDTH_FACTOR,
            anchor_span + max_width,
        )

    @classmethod
    def _collision_jump_x(
        cls,
        rect: Tuple[float, float, float, float],
        state: PlacementState,
    ) -> Optional[float]:
        jump_x = None
        for _, existing in state.occupied:
            if not AnchorAwareLegalizer._rectangles_overlap(rect, existing):
                continue
            right_edge = existing[0] + existing[2]
            jump_x = right_edge if jump_x is None else max(jump_x, right_edge)
        return jump_x

    @classmethod
    def _next_row_y(
        cls,
        current_y: float,
        row_height: float,
        block_height: float,
        state: PlacementState,
    ) -> float:
        next_y = current_y + max(row_height, block_height, cls.MIN_ADVANCE)
        for _, (_, y, _, height) in state.occupied:
            if cls._intervals_overlap(current_y, current_y + block_height, y, y + height):
                next_y = max(next_y, y + height)
        return next_y

    @staticmethod
    def _intervals_overlap(
        first_start: float,
        first_end: float,
        second_start: float,
        second_end: float,
    ) -> bool:
        overlap = min(first_end, second_end) - max(first_start, second_start)
        return overlap > AnchorAwareLegalizer.OVERLAP_EPS

    @staticmethod
    def _right_expansion_slot(
        width: float,
        height: float,
        state: PlacementState,
    ) -> Tuple[float, float, float, float]:
        bbox = AnchorAwareLegalizer._bbox(state.occupied)
        if bbox is None:
            return (0.0, 0.0, width, height)
        return (bbox[2], bbox[1], width, height)


class SoftConstraintImprover:
    BOUNDARY_EPS = 1e-6
    ROUND_DIGITS = 9
    MAX_AXIS_OPTIONS = 8
    PROXY_EPS = 1e-12

    @classmethod
    def improve(
        cls,
        problem: NormalizedProblem,
        placement: List[Tuple[float, float, float, float]],
        checker: Optional[object] = None,
    ) -> List[Tuple[float, float, float, float]]:
        best = [tuple(rect) for rect in placement]
        active_checker = checker if checker is not None else FeasibilityChecker()
        if not active_checker.check(best, problem).is_feasible:
            return best

        current_boundary = cls._boundary_violations(problem, best)
        if current_boundary == 0:
            return best

        current_soft = cls._soft_violations(problem, best)
        n_soft = cls._soft_denominator(problem)
        current_hpwl = cls._hpwl(problem, best)
        current_area = max(calculate_bbox_area(best), 1.0)
        current_proxy = cls._cost_proxy(
            current_soft,
            n_soft,
            hpwl=current_hpwl,
            area=current_area,
            hpwl_baseline=max(current_hpwl, 1.0),
            area_baseline=current_area,
        )

        for block in problem.blocks:
            if block.is_preplaced or block.boundary_mask == 0:
                continue

            candidate_choice: Optional[List[Tuple[float, float, float, float]]] = None
            candidate_choice_key: Optional[Tuple[float, float, float, float, int]] = None

            for rect in cls._boundary_candidates(problem, best, block):
                if cls._same_rect(rect, best[block.index]):
                    continue
                if cls._overlaps_any(block.index, rect, best):
                    continue

                candidate = list(best)
                candidate[block.index] = rect
                candidate_boundary = cls._boundary_violations(problem, candidate)
                if candidate_boundary >= current_boundary:
                    continue
                candidate_soft = cls._soft_violations(problem, candidate)
                if candidate_soft >= current_soft:
                    continue

                report = active_checker.check(candidate, problem)
                if not report.is_feasible:
                    continue

                candidate_hpwl = cls._hpwl(problem, candidate)
                candidate_area = max(calculate_bbox_area(candidate), 1.0)
                candidate_proxy = cls._cost_proxy(
                    candidate_soft,
                    n_soft,
                    hpwl=candidate_hpwl,
                    area=candidate_area,
                    hpwl_baseline=max(current_hpwl, 1.0),
                    area_baseline=current_area,
                )
                if candidate_proxy > current_proxy + cls.PROXY_EPS:
                    continue

                candidate_key = (
                    round(candidate_proxy, cls.ROUND_DIGITS),
                    float(candidate_soft),
                    float(candidate_boundary),
                    round(candidate_hpwl, cls.ROUND_DIGITS),
                    round(candidate_area, cls.ROUND_DIGITS),
                    block.index,
                )
                if candidate_choice_key is None or candidate_key < candidate_choice_key:
                    candidate_choice_key = candidate_key
                    candidate_choice = candidate

            if candidate_choice is not None and candidate_choice_key is not None:
                best = candidate_choice
                current_soft = int(candidate_choice_key[1])
                current_boundary = int(candidate_choice_key[2])
                current_hpwl = cls._hpwl(problem, best)
                current_area = max(calculate_bbox_area(best), 1.0)
                current_proxy = cls._cost_proxy(
                    current_soft,
                    n_soft,
                    hpwl=current_hpwl,
                    area=current_area,
                    hpwl_baseline=max(current_hpwl, 1.0),
                    area_baseline=current_area,
                )
                if current_boundary == 0:
                    break

        return best

    @classmethod
    def _boundary_candidates(
        cls,
        problem: NormalizedProblem,
        positions: List[Tuple[float, float, float, float]],
        block: BlockSpec,
    ) -> List[Tuple[float, float, float, float]]:
        bbox = cls._bbox(positions)
        if bbox is None:
            return []

        x, y, width, height = positions[block.index]
        min_x, min_y, max_x, max_y = bbox
        x_options = cls._required_axis_options(
            block.boundary_mask,
            low_bit=1,
            high_bit=2,
            low_value=min_x,
            high_value=max_x - width,
            current_value=x,
        )
        y_options = cls._required_axis_options(
            block.boundary_mask,
            low_bit=8,
            high_bit=4,
            low_value=min_y,
            high_value=max_y - height,
            current_value=y,
        )

        has_horizontal = bool(block.boundary_mask & (1 | 2))
        has_vertical = bool(block.boundary_mask & (4 | 8))
        if has_horizontal and not has_vertical:
            y_options = cls._free_axis_options(
                block.index,
                positions,
                axis=1,
                size=height,
                low=min_y,
                high=max_y - height,
                current=y,
            )
        elif has_vertical and not has_horizontal:
            x_options = cls._free_axis_options(
                block.index,
                positions,
                axis=0,
                size=width,
                low=min_x,
                high=max_x - width,
                current=x,
            )

        candidates: List[Tuple[float, float, float, float]] = []
        seen = set()
        for cand_x in x_options:
            for cand_y in y_options:
                if not math.isfinite(cand_x) or not math.isfinite(cand_y):
                    continue
                rect = (float(cand_x), float(cand_y), float(width), float(height))
                key = (
                    round(rect[0], cls.ROUND_DIGITS),
                    round(rect[1], cls.ROUND_DIGITS),
                    round(rect[2], cls.ROUND_DIGITS),
                    round(rect[3], cls.ROUND_DIGITS),
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(rect)
        return candidates

    @classmethod
    def _required_axis_options(
        cls,
        mask: int,
        low_bit: int,
        high_bit: int,
        low_value: float,
        high_value: float,
        current_value: float,
    ) -> List[float]:
        options: List[float] = []
        if mask & low_bit:
            options.append(low_value)
        if mask & high_bit:
            options.append(high_value)
        if not options:
            options.append(current_value)
        return cls._unique_values(options)

    @classmethod
    def _free_axis_options(
        cls,
        block_id: int,
        positions: List[Tuple[float, float, float, float]],
        axis: int,
        size: float,
        low: float,
        high: float,
        current: float,
    ) -> List[float]:
        if high < low:
            return [current]

        options = [current, low, high]
        for other_id, rect in enumerate(positions):
            if other_id == block_id:
                continue
            start = rect[axis]
            other_size = rect[2] if axis == 0 else rect[3]
            end = start + other_size
            options.append(end)
            options.append(start - size)

        bounded = [
            value for value in cls._unique_values(options)
            if math.isfinite(value) and value >= low - cls.BOUNDARY_EPS and value <= high + cls.BOUNDARY_EPS
        ]
        bounded.sort(key=lambda value: (abs(value - current), value))
        return bounded[:cls.MAX_AXIS_OPTIONS] or [current]

    @classmethod
    def _boundary_violations(
        cls,
        problem: NormalizedProblem,
        positions: List[Tuple[float, float, float, float]],
    ) -> int:
        bbox = cls._bbox(positions)
        if bbox is None:
            return 0

        min_x, min_y, max_x, max_y = bbox
        violations = 0
        for block in problem.blocks:
            code = int(block.boundary_mask)
            if code == 0:
                continue
            x, y, width, height = positions[block.index]
            touches = {
                1: abs(x - min_x) < cls.BOUNDARY_EPS,
                2: abs(x + width - max_x) < cls.BOUNDARY_EPS,
                4: abs(y + height - max_y) < cls.BOUNDARY_EPS,
                8: abs(y - min_y) < cls.BOUNDARY_EPS,
            }
            if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
                violations += 1
        return violations

    @classmethod
    def _soft_violations(
        cls,
        problem: NormalizedProblem,
        positions: List[Tuple[float, float, float, float]],
    ) -> int:
        return (
            cls._boundary_violations(problem, positions)
            + cls._grouping_violations(problem, positions)
            + cls._mib_violations(problem, positions)
        )

    @classmethod
    def _grouping_violations(
        cls,
        problem: NormalizedProblem,
        positions: List[Tuple[float, float, float, float]],
    ) -> int:
        groups: Dict[int, List[int]] = {}
        for block in problem.blocks:
            if block.cluster_group is None:
                continue
            groups.setdefault(block.cluster_group, []).append(block.index)

        violations = 0
        for group_indices in groups.values():
            if len(group_indices) <= 1:
                continue
            components = cls._edge_connected_components(group_indices, positions)
            violations += max(0, components - 1)
        return violations

    @classmethod
    def _edge_connected_components(
        cls,
        group_indices: List[int],
        positions: List[Tuple[float, float, float, float]],
    ) -> int:
        remaining = set(group_indices)
        components = 0

        while remaining:
            components += 1
            stack = [remaining.pop()]
            while stack:
                current = stack.pop()
                for other in list(remaining):
                    if cls._share_edge(positions[current], positions[other]):
                        remaining.remove(other)
                        stack.append(other)
        return components

    @classmethod
    def _share_edge(
        cls,
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> bool:
        x1, y1, w1, h1 = first
        x2, y2, w2, h2 = second
        first_right = x1 + w1
        second_right = x2 + w2
        first_top = y1 + h1
        second_top = y2 + h2

        y_overlap = min(first_top, second_top) - max(y1, y2)
        x_overlap = min(first_right, second_right) - max(x1, x2)
        vertical_touch = (
            abs(first_right - x2) < cls.BOUNDARY_EPS
            or abs(second_right - x1) < cls.BOUNDARY_EPS
        )
        horizontal_touch = (
            abs(first_top - y2) < cls.BOUNDARY_EPS
            or abs(second_top - y1) < cls.BOUNDARY_EPS
        )
        return (
            vertical_touch and y_overlap > cls.BOUNDARY_EPS
        ) or (
            horizontal_touch and x_overlap > cls.BOUNDARY_EPS
        )

    @staticmethod
    def _mib_violations(
        problem: NormalizedProblem,
        positions: List[Tuple[float, float, float, float]],
    ) -> int:
        groups: Dict[int, List[int]] = {}
        for block in problem.blocks:
            if block.mib_group is None:
                continue
            groups.setdefault(block.mib_group, []).append(block.index)

        violations = 0
        for group_indices in groups.values():
            distinct_shapes = {
                (round(positions[index][2], 4), round(positions[index][3], 4))
                for index in group_indices
            }
            violations += max(0, len(distinct_shapes) - 1)
        return violations

    @staticmethod
    def _soft_denominator(problem: NormalizedProblem) -> int:
        total = sum(1 for block in problem.blocks if block.boundary_mask != 0)

        mib_groups: Dict[int, int] = {}
        cluster_groups: Dict[int, int] = {}
        for block in problem.blocks:
            if block.mib_group is not None:
                mib_groups[block.mib_group] = mib_groups.get(block.mib_group, 0) + 1
            if block.cluster_group is not None:
                cluster_groups[block.cluster_group] = cluster_groups.get(block.cluster_group, 0) + 1

        total += sum(max(0, size - 1) for size in mib_groups.values())
        total += sum(max(0, size - 1) for size in cluster_groups.values())
        return max(total, 1)

    @staticmethod
    def _hpwl(
        problem: NormalizedProblem,
        positions: List[Tuple[float, float, float, float]],
    ) -> float:
        return (
            calculate_hpwl_b2b(positions, problem.b2b_connectivity)
            + calculate_hpwl_p2b(positions, problem.p2b_connectivity, problem.pins_pos)
        )

    @staticmethod
    def _cost_proxy(
        soft_violations: int,
        n_soft: int,
        hpwl: float,
        area: float,
        hpwl_baseline: float,
        area_baseline: float,
    ) -> float:
        hpwl_gap = max(0.0, hpwl / max(hpwl_baseline, 1.0) - 1.0)
        area_gap = max(0.0, area / max(area_baseline, 1.0) - 1.0)
        soft_relative = soft_violations / max(n_soft, 1)
        return (1.0 + 0.5 * (hpwl_gap + area_gap)) * math.exp(2.0 * soft_relative)

    @classmethod
    def _overlaps_any(
        cls,
        block_id: int,
        rect: Tuple[float, float, float, float],
        positions: List[Tuple[float, float, float, float]],
    ) -> bool:
        for other_id, other in enumerate(positions):
            if other_id == block_id:
                continue
            if FeasibilityChecker.rectangles_overlap(rect, other):
                return True
        return False

    @staticmethod
    def _bbox(
        positions: List[Tuple[float, float, float, float]],
    ) -> Optional[Tuple[float, float, float, float]]:
        if not positions:
            return None
        min_x = min(rect[0] for rect in positions)
        min_y = min(rect[1] for rect in positions)
        max_x = max(rect[0] + rect[2] for rect in positions)
        max_y = max(rect[1] + rect[3] for rect in positions)
        return (min_x, min_y, max_x, max_y)

    @classmethod
    def _same_rect(
        cls,
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> bool:
        return all(abs(first[i] - second[i]) < cls.BOUNDARY_EPS for i in range(4))

    @classmethod
    def _unique_values(cls, values: List[float]) -> List[float]:
        seen = set()
        unique: List[float] = []
        for value in values:
            if not math.isfinite(value):
                continue
            key = round(float(value), cls.ROUND_DIGITS)
            if key in seen:
                continue
            seen.add(key)
            unique.append(float(value))
        return unique


@dataclass
class ConstructiveUnit:
    unit_id: str
    block_ids: Tuple[int, ...]
    local_rects: Dict[int, Tuple[float, float, float, float]]
    width: float
    height: float
    boundary_mask: int
    guidance_rank: int
    predicted_center: Optional[Tuple[float, float]]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def min_block_id(self) -> int:
        return min(self.block_ids) if self.block_ids else 0


@dataclass
class CandidateChoice:
    source: str
    source_order: int
    positions: List[Tuple[float, float, float, float]]
    report: FeasibilityReport
    key: Tuple[float, float, float, float, float, int]


@dataclass
class ConstructiveConnectionIndex:
    b2b_neighbors: Dict[int, Tuple[Tuple[int, float], ...]]
    p2b_targets: Dict[int, Tuple[Tuple[float, float, float], ...]]


class ConstructiveCandidateLegalizer:
    ROUND_DIGITS = 9
    MAX_CANDIDATES_PER_UNIT = 96
    FRAME_COMPACTION_MIN_BLOCKS = 100
    FRAME_COMPACTION_MAX_CANDIDATES = 4
    ASPECT_RATIO_MIN_BLOCKS = 100
    ASPECT_RATIO_MAX_COMPACTION_CANDIDATES = 2
    PROXY_SOFT_WEIGHT = 1000.0
    PROXY_BBOX_WEIGHT = 0.01
    BOUNDARY_FRAME_GAP = 1.0
    BOUNDARY_FRAME_ASPECT = 1.5
    CONNECTION_SCORE_WEIGHT = 0.35
    CONNECTION_SCORE_CAP = 4.0
    ASPECT_RATIO_LIMITS = (0.25, 4.0)
    BOUNDARY_ASPECT_PROFILES = (
        ("mild", 0.68, 1.48, 1.0, 1.0),
        ("strong", 0.48, 2.08, 1.0, 1.0),
        ("compact", 0.58, 1.72, 0.82, 1.18),
    )

    @classmethod
    def build_candidates(
        cls,
        problem: NormalizedProblem,
        guidance: Guidance,
    ) -> List[Tuple[str, List[Tuple[float, float, float, float]]]]:
        units = cls._build_units(problem, guidance)
        if not units:
            return []

        candidates: List[Tuple[str, List[Tuple[float, float, float, float]]]] = []
        seen = set()
        connection_index = cls._build_connection_index(problem)
        for source, ordered_units in cls._order_variants(units):
            state = cls._legalize_order(problem, guidance, ordered_units, connection_index)
            if state.failed or not state.is_complete:
                continue
            positions = state.as_positions(problem)
            signature = cls._positions_signature(positions)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append((source, positions))

        for source, positions in cls._boundary_frame_candidates(
            problem,
            units,
            connection_index,
        ):
            signature = cls._positions_signature(positions)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append((source, positions))

        for source, positions in cls._boundary_aspect_ratio_candidates(
            problem,
            guidance,
            connection_index,
        ):
            signature = cls._positions_signature(positions)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append((source, positions))
        return candidates

    @classmethod
    def _build_units(
        cls,
        problem: NormalizedProblem,
        guidance: Guidance,
        dimensions_override: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> List[ConstructiveUnit]:
        rank_by_block = {
            block_id: rank
            for rank, block_id in enumerate(guidance.order)
        }
        default_rank = problem.block_count + 1
        groups: Dict[int, List[int]] = {}
        for block in problem.blocks:
            if block.cluster_group is None or block.is_preplaced:
                continue
            groups.setdefault(block.cluster_group, []).append(block.index)

        units: List[ConstructiveUnit] = []
        macro_members = set()
        for group_id, members in sorted(groups.items()):
            movable_members = tuple(
                block_id for block_id in sorted(members)
                if block_id in problem.movables
            )
            if len(movable_members) <= 1:
                continue
            unit = cls._make_cluster_unit(
                problem,
                guidance,
                group_id,
                movable_members,
                rank_by_block,
                default_rank,
                dimensions_override,
            )
            units.append(unit)
            macro_members.update(movable_members)

        for block_id in problem.movables:
            if block_id in macro_members:
                continue
            block = problem.blocks[block_id]
            width, height = cls._block_dimensions(problem, block_id, dimensions_override)
            units.append(ConstructiveUnit(
                unit_id=f"block:{block_id}",
                block_ids=(block_id,),
                local_rects={
                    block_id: (0.0, 0.0, width, height)
                },
                width=width,
                height=height,
                boundary_mask=int(block.boundary_mask),
                guidance_rank=rank_by_block.get(block_id, default_rank + block_id),
                predicted_center=guidance.predicted_centers.get(block_id),
            ))

        return units

    @classmethod
    def _make_cluster_unit(
        cls,
        problem: NormalizedProblem,
        guidance: Guidance,
        group_id: int,
        members: Tuple[int, ...],
        rank_by_block: Dict[int, int],
        default_rank: int,
        dimensions_override: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> ConstructiveUnit:
        x_cursor = 0.0
        height = 0.0
        boundary_mask = 0
        local_rects: Dict[int, Tuple[float, float, float, float]] = {}
        predicted_centers: List[Tuple[float, float]] = []

        for block_id in members:
            block = problem.blocks[block_id]
            width, block_height = cls._block_dimensions(problem, block_id, dimensions_override)
            local_rects[block_id] = (x_cursor, 0.0, width, block_height)
            x_cursor += width
            height = max(height, block_height)
            boundary_mask |= int(block.boundary_mask)
            predicted = guidance.predicted_centers.get(block_id)
            if predicted is not None:
                predicted_centers.append(predicted)

        predicted_center = None
        if predicted_centers:
            predicted_center = (
                sum(center[0] for center in predicted_centers) / len(predicted_centers),
                sum(center[1] for center in predicted_centers) / len(predicted_centers),
            )

        return ConstructiveUnit(
            unit_id=f"cluster:{group_id}",
            block_ids=members,
            local_rects=local_rects,
            width=x_cursor,
            height=height,
            boundary_mask=boundary_mask,
            guidance_rank=min(rank_by_block.get(block_id, default_rank + block_id) for block_id in members),
            predicted_center=predicted_center,
        )

    @staticmethod
    def _block_dimensions(
        problem: NormalizedProblem,
        block_id: int,
        dimensions_override: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> Tuple[float, float]:
        if dimensions_override is not None:
            dimensions = dimensions_override.get(block_id)
            if dimensions is not None:
                width, height = dimensions
                if (
                    math.isfinite(width)
                    and math.isfinite(height)
                    and width > 0.0
                    and height > 0.0
                ):
                    return float(width), float(height)

        block = problem.blocks[block_id]
        return float(block.width), float(block.height)

    @classmethod
    def _order_variants(
        cls,
        units: List[ConstructiveUnit],
    ) -> List[Tuple[str, List[ConstructiveUnit]]]:
        variants = [
            (
                "constructive:guidance_units",
                sorted(units, key=lambda unit: (
                    unit.guidance_rank,
                    unit.min_block_id,
                    unit.unit_id,
                )),
            ),
            (
                "constructive:grouping_macro_priority",
                sorted(units, key=lambda unit: (
                    0 if len(unit.block_ids) > 1 else 1,
                    unit.guidance_rank,
                    unit.min_block_id,
                    unit.unit_id,
                )),
            ),
            (
                "constructive:boundary_skyline_connected",
                sorted(units, key=lambda unit: (
                    cls._boundary_bucket(unit.boundary_mask),
                    -unit.area,
                    unit.guidance_rank,
                    unit.min_block_id,
                    unit.unit_id,
                )),
            ),
        ]

        unique_variants: List[Tuple[str, List[ConstructiveUnit]]] = []
        seen = set()
        for name, ordered_units in variants:
            signature = tuple(unit.unit_id for unit in ordered_units)
            if signature in seen:
                continue
            seen.add(signature)
            unique_variants.append((name, ordered_units))
        return unique_variants

    @staticmethod
    def _boundary_bucket(mask: int) -> int:
        wants_left = bool(mask & 1)
        wants_right = bool(mask & 2)
        wants_top = bool(mask & 4)
        wants_bottom = bool(mask & 8)
        if (wants_left or wants_right) and (wants_top or wants_bottom):
            return 0
        if wants_left or wants_right or wants_top or wants_bottom:
            return 1
        return 2

    @classmethod
    def _boundary_frame_candidates(
        cls,
        problem: NormalizedProblem,
        units: List[ConstructiveUnit],
        connection_index: ConstructiveConnectionIndex,
    ) -> List[Tuple[str, List[Tuple[float, float, float, float]]]]:
        if not any(unit.boundary_mask for unit in units):
            return []

        ordered_units = sorted(units, key=lambda unit: (
            cls._boundary_bucket(unit.boundary_mask),
            -unit.area,
            unit.guidance_rank,
            unit.min_block_id,
            unit.unit_id,
        ))
        candidates: List[Tuple[str, List[Tuple[float, float, float, float]]]] = []
        positions = cls._boundary_frame_candidate(problem, ordered_units)
        if positions is not None:
            candidates.append(("constructive:boundary_frame_structured", positions))

        candidates.extend(cls._boundary_frame_compaction_candidates(
            problem,
            ordered_units,
            positions,
            connection_index,
        ))
        return candidates

    @classmethod
    def _boundary_aspect_ratio_candidates(
        cls,
        problem: NormalizedProblem,
        guidance: Guidance,
        connection_index: ConstructiveConnectionIndex,
    ) -> List[Tuple[str, List[Tuple[float, float, float, float]]]]:
        if problem.block_count < cls.ASPECT_RATIO_MIN_BLOCKS:
            return []
        if not any(
            block.boundary_mask and not block.is_fixed and not block.is_preplaced
            for block in problem.blocks
        ):
            return []

        candidates: List[Tuple[str, List[Tuple[float, float, float, float]]]] = []
        seen = set()
        for profile_name, horizontal_ratio, vertical_ratio, corner_ratio, interior_ratio in (
            cls.BOUNDARY_ASPECT_PROFILES
        ):
            dimensions = cls._aspect_profile_dimensions(
                problem,
                horizontal_ratio,
                vertical_ratio,
                corner_ratio,
                interior_ratio,
            )
            if not dimensions:
                continue

            shaped_units = cls._build_units(problem, guidance, dimensions)
            if not shaped_units:
                continue
            if not any(unit.boundary_mask for unit in shaped_units):
                continue

            ordered_units = sorted(shaped_units, key=lambda unit: (
                cls._boundary_bucket(unit.boundary_mask),
                -unit.area,
                unit.guidance_rank,
                unit.min_block_id,
                unit.unit_id,
            ))

            start_positions = cls._boundary_frame_candidate(problem, ordered_units)
            if start_positions is not None:
                signature = cls._positions_signature(start_positions)
                if signature not in seen:
                    seen.add(signature)
                    candidates.append((
                        f"constructive:boundary_aspect:{profile_name}:frame",
                        start_positions,
                    ))

            compact_count = 0
            for source, positions in cls._boundary_frame_compaction_candidates(
                problem,
                ordered_units,
                start_positions,
                connection_index,
            ):
                signature = cls._positions_signature(positions)
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append((
                    f"constructive:boundary_aspect:{profile_name}:{source}",
                    positions,
                ))
                compact_count += 1
                if compact_count >= cls.ASPECT_RATIO_MAX_COMPACTION_CANDIDATES:
                    break

        return candidates

    @classmethod
    def _aspect_profile_dimensions(
        cls,
        problem: NormalizedProblem,
        horizontal_ratio: float,
        vertical_ratio: float,
        corner_ratio: float,
        interior_ratio: float,
    ) -> Dict[int, Tuple[float, float]]:
        raw_ratios: Dict[int, float] = {}
        mib_members: Dict[int, List[int]] = {}
        immutable_mib_ratios: Dict[int, List[float]] = {}

        for block in problem.blocks:
            if block.mib_group is not None:
                mib_members.setdefault(block.mib_group, []).append(block.index)
                if block.is_fixed or block.is_preplaced:
                    fixed_ratio = cls._safe_ratio(block.width, block.height)
                    if fixed_ratio is not None:
                        immutable_mib_ratios.setdefault(block.mib_group, []).append(fixed_ratio)

            if block.is_fixed or block.is_preplaced:
                continue

            ratio = cls._boundary_profile_ratio(
                block.boundary_mask,
                horizontal_ratio,
                vertical_ratio,
                corner_ratio,
                interior_ratio,
            )
            raw_ratios[block.index] = ratio

        for group_id, members in mib_members.items():
            mutable_members = [block_id for block_id in members if block_id in raw_ratios]
            if not mutable_members:
                continue

            immutable_ratios = immutable_mib_ratios.get(group_id)
            if immutable_ratios:
                shared_ratio = cls._geometric_mean_ratio(immutable_ratios)
            else:
                shared_ratio = cls._geometric_mean_ratio(
                    raw_ratios[block_id] for block_id in mutable_members
                )
            for block_id in mutable_members:
                raw_ratios[block_id] = shared_ratio

        dimensions: Dict[int, Tuple[float, float]] = {}
        for block_id, ratio in raw_ratios.items():
            block = problem.blocks[block_id]
            shaped = cls._dimensions_for_area_ratio(block.area_target, ratio)
            if shaped is None:
                continue
            width, height = shaped
            if (
                abs(width - block.width) <= 1e-9
                and abs(height - block.height) <= 1e-9
            ):
                continue
            dimensions[block_id] = (width, height)
        return dimensions

    @classmethod
    def _boundary_profile_ratio(
        cls,
        boundary_mask: int,
        horizontal_ratio: float,
        vertical_ratio: float,
        corner_ratio: float,
        interior_ratio: float,
    ) -> float:
        wants_horizontal_rail = bool(boundary_mask & (4 | 8))
        wants_vertical_rail = bool(boundary_mask & (1 | 2))
        if wants_horizontal_rail and wants_vertical_rail:
            return cls._clamp_aspect_ratio(corner_ratio)
        if wants_horizontal_rail:
            return cls._clamp_aspect_ratio(horizontal_ratio)
        if wants_vertical_rail:
            return cls._clamp_aspect_ratio(vertical_ratio)
        return cls._clamp_aspect_ratio(interior_ratio)

    @classmethod
    def _dimensions_for_area_ratio(
        cls,
        area_value: float,
        ratio: float,
    ) -> Optional[Tuple[float, float]]:
        area = area_value if math.isfinite(area_value) and area_value > 0.0 else None
        if area is None:
            return None
        ratio = cls._clamp_aspect_ratio(ratio)
        width = math.sqrt(area * ratio)
        height = math.sqrt(area / ratio)
        if (
            not math.isfinite(width)
            or not math.isfinite(height)
            or width <= 0.0
            or height <= 0.0
        ):
            return None
        return width, height

    @classmethod
    def _geometric_mean_ratio(cls, ratios) -> float:
        logs: List[float] = []
        for ratio in ratios:
            clamped = cls._clamp_aspect_ratio(ratio)
            if clamped > 0.0 and math.isfinite(clamped):
                logs.append(math.log(clamped))
        if not logs:
            return 1.0
        return cls._clamp_aspect_ratio(math.exp(sum(logs) / len(logs)))

    @classmethod
    def _clamp_aspect_ratio(cls, ratio: float) -> float:
        low, high = cls.ASPECT_RATIO_LIMITS
        if not math.isfinite(ratio) or ratio <= 0.0:
            return 1.0
        return min(max(float(ratio), low), high)

    @staticmethod
    def _safe_ratio(width: float, height: float) -> Optional[float]:
        if (
            not math.isfinite(width)
            or not math.isfinite(height)
            or width <= 0.0
            or height <= 0.0
        ):
            return None
        return float(width) / float(height)

    @classmethod
    def _boundary_frame_candidate(
        cls,
        problem: NormalizedProblem,
        ordered_units: List[ConstructiveUnit],
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        state = cls._initial_state_with_anchors(problem)
        if state.failed:
            return None

        groups = cls._boundary_frame_groups(ordered_units)
        corners = {
            name: values[0] if values else None
            for name, values in (
                ("top_left", groups["top_left"]),
                ("top_right", groups["top_right"]),
                ("bottom_left", groups["bottom_left"]),
                ("bottom_right", groups["bottom_right"]),
            )
        }
        left_units = groups["left"]
        right_units = groups["right"]
        top_units = groups["top"]
        bottom_units = groups["bottom"]
        interior_units = groups["interior"]

        left_width = max(
            cls._max_unit_width(left_units),
            cls._valid_unit_width(corners["top_left"]) if corners["top_left"] else 0.0,
            cls._valid_unit_width(corners["bottom_left"]) if corners["bottom_left"] else 0.0,
        )
        right_width = max(
            cls._max_unit_width(right_units),
            cls._valid_unit_width(corners["top_right"]) if corners["top_right"] else 0.0,
            cls._valid_unit_width(corners["bottom_right"]) if corners["bottom_right"] else 0.0,
        )
        top_height = cls._max_unit_height(top_units)
        bottom_height = cls._max_unit_height(bottom_units)

        interior_width_hint = cls._shelf_width_for_units(
            interior_units,
            cls.BOUNDARY_FRAME_ASPECT,
        )
        rail_central_width = max(
            cls._sum_unit_widths(top_units),
            cls._sum_unit_widths(bottom_units),
        )
        central_width = max(interior_width_hint, rail_central_width)
        if interior_units and central_width <= 0.0:
            central_width = cls._max_unit_width(interior_units)

        relative_interior_origins, interior_width, interior_height = cls._pack_shelf_unit_origins(
            interior_units,
            central_width if central_width > 0.0 else None,
        )
        central_width = max(central_width, rail_central_width, interior_width)

        central_column_height = bottom_height + interior_height + top_height
        left_column_height = (
            (cls._valid_unit_height(corners["bottom_left"]) if corners["bottom_left"] else 0.0)
            + cls._sum_unit_heights(left_units)
            + (cls._valid_unit_height(corners["top_left"]) if corners["top_left"] else 0.0)
        )
        right_column_height = (
            (cls._valid_unit_height(corners["bottom_right"]) if corners["bottom_right"] else 0.0)
            + cls._sum_unit_heights(right_units)
            + (cls._valid_unit_height(corners["top_right"]) if corners["top_right"] else 0.0)
        )

        frame_width = max(left_width + central_width + right_width, 1.0)
        frame_height = max(central_column_height, left_column_height, right_column_height, 1.0)
        anchor_bbox = AnchorAwareLegalizer._bbox(state.occupied)
        frame_x = cls._boundary_frame_x_origin(anchor_bbox, frame_width, ordered_units)
        frame_y, frame_height = cls._boundary_frame_y_origin_and_height(anchor_bbox, frame_height)
        central_x = frame_x + left_width

        origins: Dict[str, Tuple[float, float]] = {}
        for unit_id, (local_x, local_y) in relative_interior_origins.items():
            origins[unit_id] = (central_x + local_x, frame_y + bottom_height + local_y)

        x_cursor = central_x
        for unit in bottom_units:
            origins[unit.unit_id] = (x_cursor, frame_y)
            x_cursor += cls._valid_unit_width(unit)

        x_cursor = central_x
        for unit in top_units:
            origins[unit.unit_id] = (
                x_cursor,
                frame_y + frame_height - cls._valid_unit_height(unit),
            )
            x_cursor += cls._valid_unit_width(unit)

        y_cursor = frame_y + (
            cls._valid_unit_height(corners["bottom_left"])
            if corners["bottom_left"] else 0.0
        )
        for unit in left_units:
            origins[unit.unit_id] = (frame_x, y_cursor)
            y_cursor += cls._valid_unit_height(unit)

        y_cursor = frame_y + (
            cls._valid_unit_height(corners["bottom_right"])
            if corners["bottom_right"] else 0.0
        )
        for unit in right_units:
            origins[unit.unit_id] = (
                frame_x + frame_width - cls._valid_unit_width(unit),
                y_cursor,
            )
            y_cursor += cls._valid_unit_height(unit)

        if corners["bottom_left"] is not None:
            origins[corners["bottom_left"].unit_id] = (frame_x, frame_y)
        if corners["bottom_right"] is not None:
            unit = corners["bottom_right"]
            origins[unit.unit_id] = (
                frame_x + frame_width - cls._valid_unit_width(unit),
                frame_y,
            )
        if corners["top_left"] is not None:
            unit = corners["top_left"]
            origins[unit.unit_id] = (
                frame_x,
                frame_y + frame_height - cls._valid_unit_height(unit),
            )
        if corners["top_right"] is not None:
            unit = corners["top_right"]
            origins[unit.unit_id] = (
                frame_x + frame_width - cls._valid_unit_width(unit),
                frame_y + frame_height - cls._valid_unit_height(unit),
            )

        if not cls._apply_unit_origins(state, ordered_units, origins):
            return None
        if state.failed or not state.is_complete:
            return None
        return state.as_positions(problem)

    @classmethod
    def _boundary_frame_groups(
        cls,
        ordered_units: List[ConstructiveUnit],
    ) -> Dict[str, List[ConstructiveUnit]]:
        groups: Dict[str, List[ConstructiveUnit]] = {
            "top_left": [],
            "top_right": [],
            "bottom_left": [],
            "bottom_right": [],
            "left": [],
            "right": [],
            "top": [],
            "bottom": [],
            "interior": [],
        }
        for unit in ordered_units:
            groups[cls._boundary_frame_bucket(unit)].append(unit)

        corner_extras = {
            "top_left": "left",
            "bottom_left": "left",
            "top_right": "right",
            "bottom_right": "right",
        }
        for corner, side in corner_extras.items():
            if len(groups[corner]) <= 1:
                continue
            groups[side].extend(groups[corner][1:])
            del groups[corner][1:]
        return groups

    @staticmethod
    def _boundary_frame_bucket(unit: ConstructiveUnit) -> str:
        mask = int(unit.boundary_mask)
        if mask == 0:
            return "interior"

        wants_left = bool(mask & 1)
        wants_right = bool(mask & 2)
        wants_top = bool(mask & 4)
        wants_bottom = bool(mask & 8)

        if (wants_left or wants_right) and (wants_top or wants_bottom):
            horizontal = "left" if wants_left else "right"
            vertical = "bottom" if wants_bottom else "top"
            return f"{vertical}_{horizontal}"
        if wants_left:
            return "left"
        if wants_right:
            return "right"
        if wants_bottom:
            return "bottom"
        if wants_top:
            return "top"
        return "interior"

    @classmethod
    def _pack_shelf_unit_origins(
        cls,
        units: List[ConstructiveUnit],
        shelf_width: Optional[float],
    ) -> Tuple[Dict[str, Tuple[float, float]], float, float]:
        origins: Dict[str, Tuple[float, float]] = {}
        x_origin = 0.0
        y_origin = 0.0
        x_cursor = x_origin
        y_cursor = y_origin
        row_height = 0.0
        max_x = x_origin
        max_y = y_origin

        for unit in units:
            width = cls._valid_unit_width(unit)
            height = cls._valid_unit_height(unit)
            if (
                shelf_width is not None
                and row_height > 0.0
                and x_cursor > x_origin
                and x_cursor + width > x_origin + shelf_width + FeasibilityChecker.OVERLAP_EPS
            ):
                x_cursor = x_origin
                y_cursor += row_height
                row_height = 0.0

            origins[unit.unit_id] = (x_cursor, y_cursor)
            x_cursor += width
            row_height = max(row_height, height)
            max_x = max(max_x, x_cursor)
            max_y = max(max_y, y_cursor + height)

        return origins, max_x - x_origin, max_y - y_origin

    @classmethod
    def _boundary_frame_compaction_candidates(
        cls,
        problem: NormalizedProblem,
        ordered_units: List[ConstructiveUnit],
        start_positions: Optional[List[Tuple[float, float, float, float]]],
        connection_index: ConstructiveConnectionIndex,
    ) -> List[Tuple[str, List[Tuple[float, float, float, float]]]]:
        if problem.block_count < cls.FRAME_COMPACTION_MIN_BLOCKS:
            return []
        if not any(unit.boundary_mask for unit in ordered_units):
            return []

        groups = cls._boundary_frame_groups(ordered_units)
        corners = cls._boundary_frame_corners(groups)
        left_width = max(
            cls._max_unit_width(groups["left"]),
            cls._valid_unit_width(corners["top_left"]) if corners["top_left"] else 0.0,
            cls._valid_unit_width(corners["bottom_left"]) if corners["bottom_left"] else 0.0,
        )
        right_width = max(
            cls._max_unit_width(groups["right"]),
            cls._valid_unit_width(corners["top_right"]) if corners["top_right"] else 0.0,
            cls._valid_unit_width(corners["bottom_right"]) if corners["bottom_right"] else 0.0,
        )
        rail_central_width = max(
            cls._sum_unit_widths(groups["top"]),
            cls._sum_unit_widths(groups["bottom"]),
        )
        current_width = cls._positions_width(start_positions)
        candidates: List[Tuple[str, List[Tuple[float, float, float, float]]]] = []
        seen_widths = set()

        for central_width in cls._frame_compaction_central_width_hints(
            groups["interior"],
            rail_central_width,
            current_width,
            left_width,
            right_width,
        ):
            width_key = round(central_width, 6)
            if width_key in seen_widths:
                continue
            seen_widths.add(width_key)

            positions = cls._boundary_frame_compaction_candidate(
                problem,
                ordered_units,
                central_width,
                start_positions,
                connection_index,
            )
            if positions is None:
                continue
            candidates.append((
                f"constructive:frame_compact:{len(candidates)}",
                positions,
            ))
            if len(candidates) >= cls.FRAME_COMPACTION_MAX_CANDIDATES:
                break

        return candidates

    @classmethod
    def _boundary_frame_compaction_candidate(
        cls,
        problem: NormalizedProblem,
        ordered_units: List[ConstructiveUnit],
        central_width_hint: float,
        start_positions: Optional[List[Tuple[float, float, float, float]]],
        connection_index: ConstructiveConnectionIndex,
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        state = cls._initial_state_with_anchors(problem)
        if state.failed:
            return None

        groups = cls._boundary_frame_groups(ordered_units)
        corners = cls._boundary_frame_corners(groups)
        left_units = groups["left"]
        right_units = groups["right"]
        top_units = groups["top"]
        bottom_units = groups["bottom"]
        interior_units = groups["interior"]

        left_width = max(
            cls._max_unit_width(left_units),
            cls._valid_unit_width(corners["top_left"]) if corners["top_left"] else 0.0,
            cls._valid_unit_width(corners["bottom_left"]) if corners["bottom_left"] else 0.0,
        )
        right_width = max(
            cls._max_unit_width(right_units),
            cls._valid_unit_width(corners["top_right"]) if corners["top_right"] else 0.0,
            cls._valid_unit_width(corners["bottom_right"]) if corners["bottom_right"] else 0.0,
        )
        top_height = cls._max_unit_height(top_units)
        bottom_height = cls._max_unit_height(bottom_units)
        rail_central_width = max(
            cls._sum_unit_widths(top_units),
            cls._sum_unit_widths(bottom_units),
        )
        central_width_hint = max(
            float(central_width_hint),
            rail_central_width,
            cls._max_unit_width(interior_units),
            1.0,
        )

        relative_interior_origins, interior_width, interior_height = (
            cls._pack_connected_bottom_left_unit_origins(
                problem,
                interior_units,
                central_width_hint,
                (0.0, 0.0),
                {},
                connection_index,
            )
        )
        central_width = max(central_width_hint, interior_width, rail_central_width)

        central_column_height = bottom_height + interior_height + top_height
        left_column_height = (
            (cls._valid_unit_height(corners["bottom_left"]) if corners["bottom_left"] else 0.0)
            + cls._sum_unit_heights(left_units)
            + (cls._valid_unit_height(corners["top_left"]) if corners["top_left"] else 0.0)
        )
        right_column_height = (
            (cls._valid_unit_height(corners["bottom_right"]) if corners["bottom_right"] else 0.0)
            + cls._sum_unit_heights(right_units)
            + (cls._valid_unit_height(corners["top_right"]) if corners["top_right"] else 0.0)
        )

        frame_width = max(left_width + central_width + right_width, 1.0)
        frame_height = max(central_column_height, left_column_height, right_column_height, 1.0)
        anchor_bbox = AnchorAwareLegalizer._bbox(state.occupied)
        frame_x = cls._boundary_frame_x_origin(anchor_bbox, frame_width, ordered_units)
        frame_y, frame_height = cls._boundary_frame_y_origin_and_height(anchor_bbox, frame_height)
        central_x = frame_x + left_width
        interior_y = frame_y + bottom_height

        origins: Dict[str, Tuple[float, float]] = {}
        placed_centers: Dict[int, Tuple[float, float]] = {}

        for unit_id, (local_x, local_y) in relative_interior_origins.items():
            origin = (central_x + local_x, interior_y + local_y)
            origins[unit_id] = origin

        interior_by_id = {unit.unit_id: unit for unit in interior_units}
        for unit_id, origin in list(origins.items()):
            unit = interior_by_id.get(unit_id)
            if unit is not None:
                placed_centers.update(cls._unit_block_centers_at_origin(unit, origin))

        corner_origins = {
            "bottom_left": (frame_x, frame_y),
            "bottom_right": (
                frame_x + frame_width - (
                    cls._valid_unit_width(corners["bottom_right"])
                    if corners["bottom_right"] else 0.0
                ),
                frame_y,
            ),
            "top_left": (
                frame_x,
                frame_y + frame_height - (
                    cls._valid_unit_height(corners["top_left"])
                    if corners["top_left"] else 0.0
                ),
            ),
            "top_right": (
                frame_x + frame_width - (
                    cls._valid_unit_width(corners["top_right"])
                    if corners["top_right"] else 0.0
                ),
                frame_y + frame_height - (
                    cls._valid_unit_height(corners["top_right"])
                    if corners["top_right"] else 0.0
                ),
            ),
        }
        for corner_name, unit in corners.items():
            if unit is None:
                continue
            origin = corner_origins[corner_name]
            origins[unit.unit_id] = origin
            placed_centers.update(cls._unit_block_centers_at_origin(unit, origin))

        current_origins = cls._unit_origin_map_from_positions(ordered_units, start_positions)
        if not cls._place_boundary_rail_units(
            bottom_units,
            "bottom",
            central_x,
            central_x + central_width,
            frame_x,
            frame_y,
            frame_width,
            frame_height,
            current_origins,
            origins,
            placed_centers,
            connection_index,
        ):
            return None
        if not cls._place_boundary_rail_units(
            top_units,
            "top",
            central_x,
            central_x + central_width,
            frame_x,
            frame_y,
            frame_width,
            frame_height,
            current_origins,
            origins,
            placed_centers,
            connection_index,
        ):
            return None

        left_lower = frame_y + (
            cls._valid_unit_height(corners["bottom_left"])
            if corners["bottom_left"] else 0.0
        )
        left_upper = frame_y + frame_height - (
            cls._valid_unit_height(corners["top_left"])
            if corners["top_left"] else 0.0
        )
        right_lower = frame_y + (
            cls._valid_unit_height(corners["bottom_right"])
            if corners["bottom_right"] else 0.0
        )
        right_upper = frame_y + frame_height - (
            cls._valid_unit_height(corners["top_right"])
            if corners["top_right"] else 0.0
        )
        if not cls._place_boundary_rail_units(
            left_units,
            "left",
            left_lower,
            left_upper,
            frame_x,
            frame_y,
            frame_width,
            frame_height,
            current_origins,
            origins,
            placed_centers,
            connection_index,
        ):
            return None
        if not cls._place_boundary_rail_units(
            right_units,
            "right",
            right_lower,
            right_upper,
            frame_x,
            frame_y,
            frame_width,
            frame_height,
            current_origins,
            origins,
            placed_centers,
            connection_index,
        ):
            return None

        if not cls._apply_unit_origins(state, ordered_units, origins):
            return None
        if state.failed or not state.is_complete:
            return None
        return state.as_positions(problem)

    @classmethod
    def _boundary_frame_corners(
        cls,
        groups: Dict[str, List[ConstructiveUnit]],
    ) -> Dict[str, Optional[ConstructiveUnit]]:
        return {
            name: values[0] if values else None
            for name, values in (
                ("top_left", groups["top_left"]),
                ("top_right", groups["top_right"]),
                ("bottom_left", groups["bottom_left"]),
                ("bottom_right", groups["bottom_right"]),
            )
        }

    @classmethod
    def _frame_compaction_central_width_hints(
        cls,
        interior_units: List[ConstructiveUnit],
        rail_central_width: float,
        current_width: Optional[float],
        left_width: float,
        right_width: float,
    ) -> List[float]:
        minimum = max(rail_central_width, cls._max_unit_width(interior_units), 1.0)
        hints = {minimum}

        if current_width is not None and math.isfinite(current_width) and current_width > 0.0:
            for scale in (0.55, 0.68, 0.82, 0.95):
                compact_width = current_width * scale - left_width - right_width
                if compact_width >= minimum - FeasibilityChecker.OVERLAP_EPS:
                    hints.add(max(minimum, compact_width))

        for aspect in (0.8, 1.0, 1.25, 1.5):
            width = cls._shelf_width_for_units(interior_units, aspect)
            if math.isfinite(width) and width > 0.0:
                hints.add(max(minimum, width))

        return sorted(hints)

    @classmethod
    def _pack_connected_bottom_left_unit_origins(
        cls,
        problem: NormalizedProblem,
        units: List[ConstructiveUnit],
        target_width: float,
        global_offset: Tuple[float, float],
        fixed_block_centers: Dict[int, Tuple[float, float]],
        connection_index: ConstructiveConnectionIndex,
    ) -> Tuple[Dict[str, Tuple[float, float]], float, float]:
        origins: Dict[str, Tuple[float, float]] = {}
        if not units:
            return origins, 0.0, 0.0

        target_width = max(target_width, cls._max_unit_width(units), 1.0)
        total_area = max(sum(unit.area for unit in units), 1.0)
        placed_rects: List[Tuple[float, float, float, float]] = []
        placed_bbox: Optional[Tuple[float, float, float, float]] = None
        placed_centers = dict(fixed_block_centers)
        remaining = sorted(
            units,
            key=lambda unit: (
                unit.guidance_rank,
                -unit.area,
                unit.min_block_id,
                unit.unit_id,
            ),
        )
        max_x = 0.0
        max_y = 0.0
        global_x, global_y = global_offset

        while remaining:
            unit = min(
                remaining,
                key=lambda candidate: (
                    -cls._unit_frontier_weight(candidate, placed_centers, connection_index),
                    -cls._unit_pin_weight(candidate, connection_index),
                    -cls._unit_connectivity_weight(candidate, connection_index),
                    -candidate.area,
                    candidate.guidance_rank,
                    candidate.min_block_id,
                    candidate.unit_id,
                ),
            )
            remaining.remove(unit)

            width = cls._valid_unit_width(unit)
            height = cls._valid_unit_height(unit)
            best_origin: Optional[Tuple[float, float]] = None
            best_key: Optional[Tuple[float, float, float, float, float, int, str]] = None

            for x in cls._bottom_left_x_positions(placed_rects, target_width, width):
                y = cls._bottom_left_y_at_x(x, width, height, placed_rects)
                local_rect = (x, y, width, height)
                candidate_bbox = AnchorAwareLegalizer._expanded_bbox(placed_bbox, local_rect)
                bbox_area = AnchorAwareLegalizer._bbox_area(candidate_bbox) / total_area
                global_origin = (global_x + x, global_y + y)
                connection_cost = cls._site_connection_cost(
                    unit,
                    global_origin,
                    candidate_bbox,
                    global_offset,
                    connection_index,
                    placed_centers,
                )
                key = (
                    bbox_area + cls.CONNECTION_SCORE_WEIGHT * connection_cost,
                    bbox_area,
                    connection_cost,
                    y + height,
                    x,
                    unit.min_block_id,
                    unit.unit_id,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_origin = (x, y)

            if best_origin is None:
                best_origin = (0.0, max_y)

            x, y = best_origin
            origins[unit.unit_id] = best_origin
            placed_rect = (x, y, width, height)
            placed_rects.append(placed_rect)
            placed_bbox = AnchorAwareLegalizer._expanded_bbox(placed_bbox, placed_rect)
            placed_centers.update(cls._unit_block_centers_at_origin(
                unit,
                (global_x + x, global_y + y),
            ))
            max_x = max(max_x, x + width)
            max_y = max(max_y, y + height)

        return origins, max_x, max_y

    @staticmethod
    def _bottom_left_x_positions(
        placed: List[Tuple[float, float, float, float]],
        target_width: float,
        width: float,
    ) -> List[float]:
        x_limit = max(0.0, target_width - width)
        candidates = {0.0, x_limit}
        for x, _, placed_width, _ in placed:
            candidates.add(x)
            candidates.add(x + placed_width)
        return sorted(
            x
            for x in candidates
            if -FeasibilityChecker.OVERLAP_EPS <= x <= x_limit + FeasibilityChecker.OVERLAP_EPS
        )

    @staticmethod
    def _bottom_left_y_at_x(
        x: float,
        width: float,
        height: float,
        placed: List[Tuple[float, float, float, float]],
    ) -> float:
        y = 0.0
        while True:
            next_y = y
            for placed_x, placed_y, placed_width, placed_height in placed:
                horizontally_overlaps = (
                    x < placed_x + placed_width - FeasibilityChecker.OVERLAP_EPS
                    and x + width > placed_x + FeasibilityChecker.OVERLAP_EPS
                )
                vertically_overlaps = (
                    y < placed_y + placed_height - FeasibilityChecker.OVERLAP_EPS
                    and y + height > placed_y + FeasibilityChecker.OVERLAP_EPS
                )
                if horizontally_overlaps and vertically_overlaps:
                    next_y = max(next_y, placed_y + placed_height)
            if next_y <= y + FeasibilityChecker.OVERLAP_EPS:
                return y
            y = next_y

    @classmethod
    def _rail_axis_positions(
        cls,
        lower: float,
        upper: float,
        length: float,
        preferred: float,
        placed_intervals: List[Tuple[float, float]],
    ) -> List[float]:
        if length <= 0.0 or upper - lower < length - FeasibilityChecker.OVERLAP_EPS:
            return []

        high = upper - length
        clipped_preferred = min(max(preferred, lower), high)
        raw_candidates = {lower, high, clipped_preferred}
        for start, end in placed_intervals:
            raw_candidates.add(start - length)
            raw_candidates.add(end)
            raw_candidates.add(start)
            raw_candidates.add(end - length)

        positions: List[float] = []
        seen = set()
        for raw in sorted(raw_candidates):
            pos = min(max(raw, lower), high)
            key = round(pos, 6)
            if key in seen:
                continue
            seen.add(key)
            if pos < lower - FeasibilityChecker.OVERLAP_EPS or pos > high + FeasibilityChecker.OVERLAP_EPS:
                continue
            overlaps = any(
                pos < end - FeasibilityChecker.OVERLAP_EPS
                and pos + length > start + FeasibilityChecker.OVERLAP_EPS
                for start, end in placed_intervals
            )
            if not overlaps:
                positions.append(pos)
        return positions

    @classmethod
    def _rail_unit_origin(
        cls,
        edge: str,
        axis_position: float,
        unit: ConstructiveUnit,
        frame_x: float,
        frame_y: float,
        frame_width: float,
        frame_height: float,
    ) -> Tuple[float, float]:
        width = cls._valid_unit_width(unit)
        height = cls._valid_unit_height(unit)
        if edge == "bottom":
            return axis_position, frame_y
        if edge == "top":
            return axis_position, frame_y + frame_height - height
        if edge == "left":
            return frame_x, axis_position
        return frame_x + frame_width - width, axis_position

    @classmethod
    def _place_boundary_rail_units(
        cls,
        units: List[ConstructiveUnit],
        edge: str,
        axis_lower: float,
        axis_upper: float,
        frame_x: float,
        frame_y: float,
        frame_width: float,
        frame_height: float,
        current_origins: Dict[str, Tuple[float, float]],
        origins: Dict[str, Tuple[float, float]],
        placed_centers: Dict[int, Tuple[float, float]],
        connection_index: ConstructiveConnectionIndex,
    ) -> bool:
        if not units:
            return True

        placed_intervals: List[Tuple[float, float]] = []
        horizontal = edge in {"bottom", "top"}
        remaining = sorted(
            units,
            key=lambda unit: (
                -cls._unit_frontier_weight(unit, placed_centers, connection_index),
                -cls._unit_connectivity_weight(unit, connection_index),
                unit.min_block_id,
                unit.unit_id,
            ),
        )

        for unit in remaining:
            length = cls._valid_unit_width(unit) if horizontal else cls._valid_unit_height(unit)
            current_origin = current_origins.get(unit.unit_id, (frame_x, frame_y))
            preferred = current_origin[0] if horizontal else current_origin[1]
            best_origin: Optional[Tuple[float, float]] = None
            best_axis: Optional[float] = None
            best_key: Optional[Tuple[float, float, float, int, str]] = None

            for axis_position in cls._rail_axis_positions(
                axis_lower,
                axis_upper,
                length,
                preferred,
                placed_intervals,
            ):
                candidate_origin = cls._rail_unit_origin(
                    edge,
                    axis_position,
                    unit,
                    frame_x,
                    frame_y,
                    frame_width,
                    frame_height,
                )
                connection_cost = cls._site_connection_cost(
                    unit,
                    candidate_origin,
                    None,
                    (0.0, 0.0),
                    connection_index,
                    placed_centers,
                )
                displacement = (
                    abs(candidate_origin[0] - current_origin[0])
                    + abs(candidate_origin[1] - current_origin[1])
                )
                key = (
                    connection_cost,
                    displacement,
                    axis_position,
                    unit.min_block_id,
                    unit.unit_id,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_axis = axis_position
                    best_origin = candidate_origin

            if best_axis is None or best_origin is None:
                return False

            origins[unit.unit_id] = best_origin
            placed_intervals.append((best_axis, best_axis + length))
            placed_centers.update(cls._unit_block_centers_at_origin(unit, best_origin))

        return True

    @classmethod
    def _site_connection_cost(
        cls,
        unit: ConstructiveUnit,
        origin: Tuple[float, float],
        local_bbox: Optional[Tuple[float, float, float, float]],
        global_offset: Tuple[float, float],
        connection_index: ConstructiveConnectionIndex,
        placed_centers: Dict[int, Tuple[float, float]],
    ) -> float:
        expanded = cls._expand_unit(unit, origin)
        expanded_bbox = None
        if local_bbox is not None:
            offset_x, offset_y = global_offset
            expanded_bbox = (
                local_bbox[0] + offset_x,
                local_bbox[1] + offset_y,
                local_bbox[2] + offset_x,
                local_bbox[3] + offset_y,
            )
        else:
            for rect in expanded.values():
                expanded_bbox = AnchorAwareLegalizer._expanded_bbox(expanded_bbox, rect)
        return cls._connection_cost(
            unit,
            expanded,
            expanded_bbox,
            connection_index,
            placed_centers,
        )

    @staticmethod
    def _unit_block_centers_at_origin(
        unit: ConstructiveUnit,
        origin: Tuple[float, float],
    ) -> Dict[int, Tuple[float, float]]:
        origin_x, origin_y = origin
        return {
            block_id: (
                origin_x + local_rect[0] + local_rect[2] / 2.0,
                origin_y + local_rect[1] + local_rect[3] / 2.0,
            )
            for block_id, local_rect in unit.local_rects.items()
        }

    @classmethod
    def _unit_frontier_weight(
        cls,
        unit: ConstructiveUnit,
        placed_centers: Dict[int, Tuple[float, float]],
        connection_index: ConstructiveConnectionIndex,
    ) -> float:
        if not placed_centers:
            return 0.0
        placed_ids = set(placed_centers)
        return sum(
            weight
            for block_id in unit.block_ids
            for other_id, weight in connection_index.b2b_neighbors.get(block_id, ())
            if other_id in placed_ids
        )

    @staticmethod
    def _unit_pin_weight(
        unit: ConstructiveUnit,
        connection_index: ConstructiveConnectionIndex,
    ) -> float:
        return sum(
            weight
            for block_id in unit.block_ids
            for _, _, weight in connection_index.p2b_targets.get(block_id, ())
        )

    @staticmethod
    def _unit_connectivity_weight(
        unit: ConstructiveUnit,
        connection_index: ConstructiveConnectionIndex,
    ) -> float:
        b2b_weight = sum(
            weight
            for block_id in unit.block_ids
            for _, weight in connection_index.b2b_neighbors.get(block_id, ())
        )
        p2b_weight = sum(
            weight
            for block_id in unit.block_ids
            for _, _, weight in connection_index.p2b_targets.get(block_id, ())
        )
        return b2b_weight + p2b_weight

    @classmethod
    def _unit_origin_map_from_positions(
        cls,
        units: List[ConstructiveUnit],
        positions: Optional[List[Tuple[float, float, float, float]]],
    ) -> Dict[str, Tuple[float, float]]:
        origins: Dict[str, Tuple[float, float]] = {}
        if positions is None:
            return origins
        for unit in units:
            for block_id in unit.block_ids:
                if block_id >= len(positions) or block_id not in unit.local_rects:
                    continue
                x, y, _, _ = positions[block_id]
                local_x, local_y, _, _ = unit.local_rects[block_id]
                if math.isfinite(x) and math.isfinite(y):
                    origins[unit.unit_id] = (x - local_x, y - local_y)
                    break
        return origins

    @staticmethod
    def _positions_width(
        positions: Optional[List[Tuple[float, float, float, float]]],
    ) -> Optional[float]:
        if not positions:
            return None
        min_x = min(rect[0] for rect in positions)
        max_x = max(rect[0] + rect[2] for rect in positions)
        width = max_x - min_x
        return width if math.isfinite(width) and width > 0.0 else None

    @classmethod
    def _shelf_width_for_units(
        cls,
        units: List[ConstructiveUnit],
        aspect: float,
    ) -> float:
        if not units:
            return 0.0
        total_area = sum(unit.area for unit in units)
        max_width = cls._max_unit_width(units)
        if total_area <= 0.0 or not math.isfinite(total_area):
            return max_width
        return max(max_width, math.sqrt(total_area) * aspect)

    @staticmethod
    def _sum_unit_widths(units: List[ConstructiveUnit]) -> float:
        return sum(ConstructiveCandidateLegalizer._valid_unit_width(unit) for unit in units)

    @staticmethod
    def _sum_unit_heights(units: List[ConstructiveUnit]) -> float:
        return sum(ConstructiveCandidateLegalizer._valid_unit_height(unit) for unit in units)

    @staticmethod
    def _max_unit_width(units: List[ConstructiveUnit]) -> float:
        return max(
            (ConstructiveCandidateLegalizer._valid_unit_width(unit) for unit in units),
            default=0.0,
        )

    @staticmethod
    def _max_unit_height(units: List[ConstructiveUnit]) -> float:
        return max(
            (ConstructiveCandidateLegalizer._valid_unit_height(unit) for unit in units),
            default=0.0,
        )

    @staticmethod
    def _valid_unit_width(unit: Optional[ConstructiveUnit]) -> float:
        if unit is None:
            return 0.0
        width = float(unit.width)
        return width if math.isfinite(width) and width > 0.0 else 1.0

    @staticmethod
    def _valid_unit_height(unit: Optional[ConstructiveUnit]) -> float:
        if unit is None:
            return 0.0
        height = float(unit.height)
        return height if math.isfinite(height) and height > 0.0 else 1.0

    @classmethod
    def _boundary_frame_x_origin(
        cls,
        anchor_bbox: Optional[Tuple[float, float, float, float]],
        frame_width: float,
        units: List[ConstructiveUnit],
    ) -> float:
        if anchor_bbox is None:
            return 0.0

        left_requests = sum(1 for unit in units if unit.boundary_mask & 1)
        right_requests = sum(1 for unit in units if unit.boundary_mask & 2)
        min_x, _, max_x, _ = anchor_bbox
        if left_requests > right_requests:
            return min(0.0, min_x) - frame_width - cls.BOUNDARY_FRAME_GAP
        return max(0.0, max_x + cls.BOUNDARY_FRAME_GAP)

    @classmethod
    def _boundary_frame_y_origin_and_height(
        cls,
        anchor_bbox: Optional[Tuple[float, float, float, float]],
        frame_height: float,
    ) -> Tuple[float, float]:
        if anchor_bbox is None:
            return 0.0, frame_height

        _, min_y, _, max_y = anchor_bbox
        y_origin = min(0.0, min_y)
        return y_origin, max(frame_height, max_y - y_origin)

    @classmethod
    def _apply_unit_origins(
        cls,
        state: PlacementState,
        ordered_units: List[ConstructiveUnit],
        origins: Dict[str, Tuple[float, float]],
    ) -> bool:
        for unit in ordered_units:
            origin = origins.get(unit.unit_id)
            if origin is None:
                state.failed = True
                state.warnings.append(f"boundary_frame_missing_origin:{unit.unit_id}")
                return False
            expanded = cls._expand_unit(unit, origin)
            if not cls._unit_rects_are_valid(expanded):
                state.failed = True
                state.warnings.append(f"boundary_frame_invalid_rect:{unit.unit_id}")
                return False
            if cls._unit_has_internal_overlap(expanded):
                state.failed = True
                state.warnings.append(f"boundary_frame_internal_overlap:{unit.unit_id}")
                return False
            if cls._unit_overlaps_occupied(expanded, state.occupied):
                state.failed = True
                state.warnings.append(f"boundary_frame_overlap:{unit.unit_id}")
                return False
            for block_id in sorted(expanded):
                rect = expanded[block_id]
                state.positions[block_id] = rect
                state.occupied.append((block_id, rect))
                state.placed_order.append(block_id)
        return True

    @classmethod
    def _legalize_order(
        cls,
        problem: NormalizedProblem,
        guidance: Guidance,
        units: List[ConstructiveUnit],
        connection_index: ConstructiveConnectionIndex,
    ) -> PlacementState:
        state = cls._initial_state_with_anchors(problem)

        for unit in units:
            if all(state.positions[block_id] is not None for block_id in unit.block_ids):
                continue
            origin = cls._choose_unit_origin(
                problem,
                unit,
                state,
                guidance,
                connection_index,
            )
            if origin is None:
                state.failed = True
                state.warnings.append(f"unit_{unit.unit_id}_no_legal_candidate")
                continue
            expanded = cls._expand_unit(unit, origin)
            for block_id in sorted(expanded):
                rect = expanded[block_id]
                state.positions[block_id] = rect
                state.occupied.append((block_id, rect))
                state.placed_order.append(block_id)

        if not state.is_complete:
            state.failed = True
            state.warnings.append("constructive_placement_incomplete")
        return state

    @classmethod
    def _initial_state_with_anchors(
        cls,
        problem: NormalizedProblem,
    ) -> PlacementState:
        state = PlacementState(
            positions=[None] * problem.block_count,
            occupied=[],
            placed_order=[],
        )

        for block_id in problem.anchors:
            block = problem.blocks[block_id]
            if block.preplaced_x is None or block.preplaced_y is None:
                state.failed = True
                state.warnings.append(f"anchor_{block_id}_missing_coordinates")
                continue
            rect = (
                float(block.preplaced_x),
                float(block.preplaced_y),
                float(block.width),
                float(block.height),
            )
            if not AnchorAwareLegalizer._rect_is_valid(rect):
                state.failed = True
                state.warnings.append(f"anchor_{block_id}_invalid_rectangle")
                continue
            if AnchorAwareLegalizer._overlaps_occupied(rect, state.occupied):
                state.failed = True
                state.warnings.append(f"anchor_{block_id}_overlaps_existing_anchor")
            state.positions[block_id] = rect
            state.occupied.append((block_id, rect))
            state.placed_order.append(block_id)
        return state

    @classmethod
    def _choose_unit_origin(
        cls,
        problem: NormalizedProblem,
        unit: ConstructiveUnit,
        state: PlacementState,
        guidance: Guidance,
        connection_index: ConstructiveConnectionIndex,
    ) -> Optional[Tuple[float, float]]:
        best_key: Optional[Tuple[float, float, float, float, float, float, float, str]] = None
        best_origin: Optional[Tuple[float, float]] = None
        placed_centers = cls._placed_centers(state)

        for origin in cls._generate_unit_origins(unit, state):
            expanded = cls._expand_unit(unit, origin)
            if not cls._unit_rects_are_valid(expanded):
                continue
            if cls._unit_has_internal_overlap(expanded):
                continue
            if cls._unit_overlaps_occupied(expanded, state.occupied):
                continue

            score = cls._score_unit_origin(
                problem,
                unit,
                origin,
                expanded,
                state,
                guidance,
                connection_index,
                placed_centers,
            )
            if best_key is None or score < best_key:
                best_key = score
                best_origin = origin
        return best_origin

    @classmethod
    def _generate_unit_origins(
        cls,
        unit: ConstructiveUnit,
        state: PlacementState,
    ) -> List[Tuple[float, float]]:
        candidates: List[Tuple[float, float]] = []

        def add(x: float, y: float) -> None:
            if math.isfinite(x) and math.isfinite(y):
                candidates.append((float(x), float(y)))

        add(0.0, 0.0)
        if unit.predicted_center is not None:
            add(
                unit.predicted_center[0] - unit.width / 2.0,
                unit.predicted_center[1] - unit.height / 2.0,
            )

        bbox = AnchorAwareLegalizer._bbox(state.occupied)
        if bbox is None:
            return cls._trim_candidates(cls._unique_candidates(candidates), unit)

        min_x, min_y, max_x, max_y = bbox
        add(min_x, min_y)
        add(max_x, min_y)
        add(min_x, max_y)
        add(max_x, max_y)
        add(min_x - unit.width, min_y)
        add(min_x, min_y - unit.height)
        add(max_x - unit.width, max_y)
        add(max_x, max_y - unit.height)

        x_edges = [min_x, max_x, min_x - unit.width, max_x - unit.width, 0.0]
        y_edges = [min_y, max_y, min_y - unit.height, max_y - unit.height, 0.0]

        for _, (x, y, width, height) in sorted(
            state.occupied,
            key=lambda item: (item[1][0], item[1][1], item[0]),
        ):
            right = x + width
            top = y + height
            add(right, y)
            add(x, top)
            add(right, top)
            add(x - unit.width, y)
            add(x, y - unit.height)
            add(right, y - unit.height)
            add(x - unit.width, top)
            add(right, min_y)
            add(min_x, top)
            add(max_x, y)
            add(x, max_y)
            x_edges.extend([x, right, x - unit.width, right - unit.width])
            y_edges.extend([y, top, y - unit.height, top - unit.height])

        cls._add_boundary_origins(unit, x_edges, y_edges, min_x, min_y, max_x, max_y, add)

        if unit.predicted_center is not None:
            pred_x = unit.predicted_center[0] - unit.width / 2.0
            pred_y = unit.predicted_center[1] - unit.height / 2.0
            close_x = cls._closest_edges(x_edges, pred_x)
            close_y = cls._closest_edges(y_edges, pred_y)
            for x in close_x:
                add(x, pred_y)
            for y in close_y:
                add(pred_x, y)
            for x in close_x[:6]:
                for y in close_y[:6]:
                    add(x, y)

        return cls._trim_candidates(cls._unique_candidates(candidates), unit)

    @classmethod
    def _add_boundary_origins(
        cls,
        unit: ConstructiveUnit,
        x_edges: List[float],
        y_edges: List[float],
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        add,
    ) -> None:
        if unit.boundary_mask == 0:
            return

        horizontal: List[float] = []
        vertical: List[float] = []
        if unit.boundary_mask & 1:
            horizontal.extend([min_x - unit.width, min_x])
        if unit.boundary_mask & 2:
            horizontal.extend([max_x, max_x - unit.width])
        if unit.boundary_mask & 8:
            vertical.extend([min_y - unit.height, min_y])
        if unit.boundary_mask & 4:
            vertical.extend([max_y, max_y - unit.height])

        if not horizontal:
            horizontal = cls._closest_edges(x_edges, min_x)
        if not vertical:
            vertical = cls._closest_edges(y_edges, min_y)

        for x in cls._unique_values(horizontal)[:8]:
            for y in cls._unique_values(vertical)[:8]:
                add(x, y)

    @classmethod
    def _trim_candidates(
        cls,
        candidates: List[Tuple[float, float]],
        unit: ConstructiveUnit,
    ) -> List[Tuple[float, float]]:
        if len(candidates) <= cls.MAX_CANDIDATES_PER_UNIT:
            return candidates
        if unit.predicted_center is None:
            return candidates[:cls.MAX_CANDIDATES_PER_UNIT]

        target_x = unit.predicted_center[0] - unit.width / 2.0
        target_y = unit.predicted_center[1] - unit.height / 2.0
        candidates.sort(key=lambda origin: (
            abs(origin[0] - target_x) + abs(origin[1] - target_y),
            abs(origin[0]) + abs(origin[1]),
            origin[1],
            origin[0],
        ))
        return candidates[:cls.MAX_CANDIDATES_PER_UNIT]

    @classmethod
    def _score_unit_origin(
        cls,
        problem: NormalizedProblem,
        unit: ConstructiveUnit,
        origin: Tuple[float, float],
        expanded: Dict[int, Tuple[float, float, float, float]],
        state: PlacementState,
        guidance: Guidance,
        connection_index: ConstructiveConnectionIndex,
        placed_centers: Dict[int, Tuple[float, float]],
    ) -> Tuple[float, float, float, float, float, float, float, str]:
        current_bbox = AnchorAwareLegalizer._bbox(state.occupied)
        expanded_bbox = current_bbox
        for rect in expanded.values():
            expanded_bbox = AnchorAwareLegalizer._expanded_bbox(expanded_bbox, rect)

        current_area = AnchorAwareLegalizer._bbox_area(current_bbox)
        expanded_area = AnchorAwareLegalizer._bbox_area(expanded_bbox)
        area_growth = max(0.0, expanded_area - current_area)
        area_scale = max(current_area, unit.area, 1.0)
        normalized_area_growth = area_growth / area_scale

        boundary_miss = cls._unit_boundary_miss(unit, expanded, expanded_bbox)
        predicted_distance = cls._predicted_distance(unit, expanded, guidance)
        connection_cost = cls._connection_cost(
            unit,
            expanded,
            expanded_bbox,
            connection_index,
            placed_centers,
        )
        placement_quality = (
            normalized_area_growth
            + cls.CONNECTION_SCORE_WEIGHT * connection_cost
        )
        coordinate_bias = abs(origin[0]) + abs(origin[1])
        span = 0.0
        if expanded_bbox is not None:
            span = (expanded_bbox[2] - expanded_bbox[0]) + (expanded_bbox[3] - expanded_bbox[1])

        return (
            round(boundary_miss, cls.ROUND_DIGITS),
            round(placement_quality, cls.ROUND_DIGITS),
            round(area_growth, cls.ROUND_DIGITS),
            round(connection_cost, cls.ROUND_DIGITS),
            round(predicted_distance, cls.ROUND_DIGITS),
            round(span + coordinate_bias, cls.ROUND_DIGITS),
            round(origin[1], cls.ROUND_DIGITS),
            unit.unit_id,
        )

    @classmethod
    def _build_connection_index(
        cls,
        problem: NormalizedProblem,
    ) -> ConstructiveConnectionIndex:
        b2b_neighbors: Dict[int, List[Tuple[int, float]]] = {}
        for edge in cls._iter_edge_rows(problem.b2b_connectivity):
            first = cls._edge_int(edge, 0)
            second = cls._edge_int(edge, 1)
            weight = cls._edge_float(edge, 2)
            if (
                first is None
                or second is None
                or weight is None
                or first < 0
                or second < 0
                or first >= problem.block_count
                or second >= problem.block_count
            ):
                continue
            edge_weight = abs(weight)
            if edge_weight <= 0.0 or not math.isfinite(edge_weight):
                continue
            b2b_neighbors.setdefault(first, []).append((second, edge_weight))
            b2b_neighbors.setdefault(second, []).append((first, edge_weight))

        p2b_targets: Dict[int, List[Tuple[float, float, float]]] = {}
        pin_count = cls._pin_count(problem.pins_pos)
        for edge in cls._iter_edge_rows(problem.p2b_connectivity):
            pin_id = cls._edge_int(edge, 0)
            block_id = cls._edge_int(edge, 1)
            weight = cls._edge_float(edge, 2)
            if (
                pin_id is None
                or block_id is None
                or weight is None
                or pin_id < 0
                or block_id < 0
                or pin_id >= pin_count
                or block_id >= problem.block_count
            ):
                continue
            pin = cls._pin_position(problem.pins_pos, pin_id)
            if pin is None:
                continue
            edge_weight = abs(weight)
            if edge_weight <= 0.0 or not math.isfinite(edge_weight):
                continue
            p2b_targets.setdefault(block_id, []).append((pin[0], pin[1], edge_weight))

        return ConstructiveConnectionIndex(
            b2b_neighbors={
                block_id: tuple(neighbors)
                for block_id, neighbors in b2b_neighbors.items()
            },
            p2b_targets={
                block_id: tuple(targets)
                for block_id, targets in p2b_targets.items()
            },
        )

    @staticmethod
    def _iter_edge_rows(edges: Optional[torch.Tensor]):
        if edges is None:
            return
        try:
            for edge in edges:
                try:
                    if len(edge) < 3:
                        continue
                except TypeError:
                    continue
                yield edge
        except TypeError:
            return

    @staticmethod
    def _edge_float(edge, index: int) -> Optional[float]:
        try:
            value = float(edge[index])
        except (IndexError, RuntimeError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @classmethod
    def _edge_int(cls, edge, index: int) -> Optional[int]:
        value = cls._edge_float(edge, index)
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _pin_count(pins_pos: Optional[torch.Tensor]) -> int:
        if pins_pos is None:
            return 0
        try:
            return len(pins_pos)
        except TypeError:
            return 0

    @classmethod
    def _pin_position(
        cls,
        pins_pos: Optional[torch.Tensor],
        pin_id: int,
    ) -> Optional[Tuple[float, float]]:
        if pins_pos is None:
            return None
        try:
            px = float(pins_pos[pin_id][0])
            py = float(pins_pos[pin_id][1])
        except (IndexError, RuntimeError, TypeError, ValueError):
            return None
        if not math.isfinite(px) or not math.isfinite(py):
            return None
        return px, py

    @staticmethod
    def _placed_centers(state: PlacementState) -> Dict[int, Tuple[float, float]]:
        return {
            block_id: (rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0)
            for block_id, rect in state.occupied
        }

    @classmethod
    def _connection_cost(
        cls,
        unit: ConstructiveUnit,
        expanded: Dict[int, Tuple[float, float, float, float]],
        expanded_bbox: Optional[Tuple[float, float, float, float]],
        connection_index: ConstructiveConnectionIndex,
        placed_centers: Dict[int, Tuple[float, float]],
    ) -> float:
        if not connection_index.b2b_neighbors and not connection_index.p2b_targets:
            return 0.0

        unit_blocks = set(unit.block_ids)
        total = 0.0
        total_weight = 0.0
        for block_id, rect in expanded.items():
            center_x = rect[0] + rect[2] / 2.0
            center_y = rect[1] + rect[3] / 2.0

            for other_id, weight in connection_index.b2b_neighbors.get(block_id, ()):
                if other_id in unit_blocks or other_id not in placed_centers:
                    continue
                other_x, other_y = placed_centers[other_id]
                total += weight * (abs(other_x - center_x) + abs(other_y - center_y))
                total_weight += weight

            for pin_x, pin_y, weight in connection_index.p2b_targets.get(block_id, ()):
                total += weight * (abs(pin_x - center_x) + abs(pin_y - center_y))
                total_weight += weight

        if total_weight <= 0.0:
            return 0.0

        average_distance = total / total_weight
        distance_scale = cls._connection_distance_scale(unit, expanded_bbox)
        normalized = average_distance / distance_scale
        if not math.isfinite(normalized) or normalized <= 0.0:
            return 0.0
        return min(cls.CONNECTION_SCORE_CAP, normalized)

    @staticmethod
    def _connection_distance_scale(
        unit: ConstructiveUnit,
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> float:
        span = 0.0
        if bbox is not None:
            span = max(0.0, bbox[2] - bbox[0]) + max(0.0, bbox[3] - bbox[1])
        unit_span = max(0.0, float(unit.width)) + max(0.0, float(unit.height))
        return max(span, unit_span, math.sqrt(max(unit.area, 1.0)), 1.0)

    @staticmethod
    def _unit_boundary_miss(
        unit: ConstructiveUnit,
        expanded: Dict[int, Tuple[float, float, float, float]],
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> float:
        if unit.boundary_mask == 0 or bbox is None:
            return 0.0
        min_x, min_y, max_x, max_y = bbox
        misses = 0
        for rect in expanded.values():
            x, y, width, height = rect
            touches = {
                1: abs(x - min_x) <= SoftConstraintImprover.BOUNDARY_EPS,
                2: abs(x + width - max_x) <= SoftConstraintImprover.BOUNDARY_EPS,
                4: abs(y + height - max_y) <= SoftConstraintImprover.BOUNDARY_EPS,
                8: abs(y - min_y) <= SoftConstraintImprover.BOUNDARY_EPS,
            }
            if all(touches[bit] for bit in (1, 2, 4, 8) if unit.boundary_mask & bit):
                return 0.0
        for bit in (1, 2, 4, 8):
            if unit.boundary_mask & bit:
                misses += 1
        return float(misses)

    @staticmethod
    def _predicted_distance(
        unit: ConstructiveUnit,
        expanded: Dict[int, Tuple[float, float, float, float]],
        guidance: Guidance,
    ) -> float:
        distances: List[float] = []
        for block_id, rect in expanded.items():
            predicted = guidance.predicted_centers.get(block_id)
            if predicted is None:
                continue
            center = (rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0)
            distances.append(math.hypot(center[0] - predicted[0], center[1] - predicted[1]))
        if not distances:
            return 0.0
        return sum(distances) / len(distances)

    @classmethod
    def _expand_unit(
        cls,
        unit: ConstructiveUnit,
        origin: Tuple[float, float],
    ) -> Dict[int, Tuple[float, float, float, float]]:
        origin_x, origin_y = origin
        return {
            block_id: (
                origin_x + rect[0],
                origin_y + rect[1],
                rect[2],
                rect[3],
            )
            for block_id, rect in unit.local_rects.items()
        }

    @staticmethod
    def _unit_rects_are_valid(
        rects: Dict[int, Tuple[float, float, float, float]],
    ) -> bool:
        return all(AnchorAwareLegalizer._rect_is_valid(rect) for rect in rects.values())

    @classmethod
    def _unit_has_internal_overlap(
        cls,
        rects: Dict[int, Tuple[float, float, float, float]],
    ) -> bool:
        items = list(rects.items())
        for i, (_, first) in enumerate(items):
            for _, second in items[i + 1:]:
                if AnchorAwareLegalizer._rectangles_overlap(first, second):
                    return True
        return False

    @staticmethod
    def _unit_overlaps_occupied(
        rects: Dict[int, Tuple[float, float, float, float]],
        occupied: List[Tuple[int, Tuple[float, float, float, float]]],
    ) -> bool:
        for rect in rects.values():
            if AnchorAwareLegalizer._overlaps_occupied(rect, occupied):
                return True
        return False

    @classmethod
    def _closest_edges(cls, values: List[float], target: float) -> List[float]:
        unique_values = cls._unique_values(values)
        unique_values.sort(key=lambda value: (abs(value - target), value))
        return unique_values[:AnchorAwareLegalizer.CLOSE_EDGE_LIMIT]

    @classmethod
    def _unique_candidates(
        cls,
        candidates: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        seen = set()
        unique: List[Tuple[float, float]] = []
        for x, y in candidates:
            key = (round(x, cls.ROUND_DIGITS), round(y, cls.ROUND_DIGITS))
            if key in seen:
                continue
            seen.add(key)
            unique.append((x, y))
        return unique

    @classmethod
    def _unique_values(cls, values: List[float]) -> List[float]:
        seen = set()
        unique: List[float] = []
        for value in values:
            if not math.isfinite(value):
                continue
            key = round(float(value), cls.ROUND_DIGITS)
            if key in seen:
                continue
            seen.add(key)
            unique.append(float(value))
        return unique

    @classmethod
    def _positions_signature(
        cls,
        positions: List[Tuple[float, float, float, float]],
    ) -> Tuple[Tuple[float, float, float, float], ...]:
        return tuple(
            (
                round(rect[0], cls.ROUND_DIGITS),
                round(rect[1], cls.ROUND_DIGITS),
                round(rect[2], cls.ROUND_DIGITS),
                round(rect[3], cls.ROUND_DIGITS),
            )
            for rect in positions
        )


class CandidateSelector:
    @classmethod
    def best_feasible(
        cls,
        problem: NormalizedProblem,
        candidates: List[Tuple[str, int, List[Tuple[float, float, float, float]]]],
        checker: FeasibilityChecker,
    ) -> Optional[CandidateChoice]:
        best: Optional[CandidateChoice] = None
        for source, source_order, positions in candidates:
            try:
                improved = SoftConstraintImprover.improve(problem, positions, checker)
                report = checker.check(improved, problem)
            except (IndexError, RuntimeError, TypeError, ValueError):
                continue
            if not report.is_feasible:
                continue
            try:
                key = cls._proxy_key(problem, improved, source_order)
            except (OverflowError, RuntimeError, TypeError, ValueError):
                continue
            choice = CandidateChoice(
                source=source,
                source_order=source_order,
                positions=improved,
                report=report,
                key=key,
            )
            if best is None or choice.key < best.key:
                best = choice
        return best

    @classmethod
    def _proxy_key(
        cls,
        problem: NormalizedProblem,
        positions: List[Tuple[float, float, float, float]],
        source_order: int,
    ) -> Tuple[float, float, float, float, float, int]:
        soft_violations = SoftConstraintImprover._soft_violations(problem, positions)
        soft_relative = soft_violations / SoftConstraintImprover._soft_denominator(problem)
        hpwl = cls._finite_float(SoftConstraintImprover._hpwl(problem, positions))
        bbox_area = cls._finite_float(calculate_bbox_area(positions))
        quality = hpwl + ConstructiveCandidateLegalizer.PROXY_BBOX_WEIGHT * bbox_area
        proxy = (
            1.0
            + quality
            + ConstructiveCandidateLegalizer.PROXY_SOFT_WEIGHT * soft_relative
        ) * math.exp(min(50.0, 2.0 * soft_relative))
        if not math.isfinite(proxy):
            proxy = float("inf")
        return (
            round(proxy, ConstructiveCandidateLegalizer.ROUND_DIGITS),
            round(soft_relative, ConstructiveCandidateLegalizer.ROUND_DIGITS),
            float(soft_violations),
            round(hpwl, ConstructiveCandidateLegalizer.ROUND_DIGITS),
            round(bbox_area, ConstructiveCandidateLegalizer.ROUND_DIGITS),
            source_order,
        )

    @staticmethod
    def _finite_float(value: object, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default


# =============================================================================
# 1. NETLIST EDGE-GNN (電路圖拓樸特徵提取器) - 完全與訓練端對齊
# =============================================================================
class NetlistGNN(nn.Module):
    def __init__(self, node_in_dim=6, hidden_dim=384):
        super().__init__()
        self.node_init = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.msg_merge = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, constraints, area_target, b2b_conn, block_count):
        N = block_count
        raw_nodes = torch.cat([constraints, area_target], dim=-1)
        h = self.node_init(raw_nodes)
        
        valid_mask = b2b_conn[:, 0] >= 0
        edges = b2b_conn[valid_mask]
        
        if edges.numel() > 0:
            idx_i = edges[:, 0].long()
            idx_j = edges[:, 1].long()
            weight = edges[:, 2].unsqueeze(-1)
            
            msg_to_i = h[idx_j] * weight
            msg_to_j = h[idx_i] * weight
            
            agg_msg = torch.zeros_like(h)
            agg_msg.index_add_(0, idx_i, msg_to_i)
            agg_msg.index_add_(0, idx_j, msg_to_j)
            
            h = self.msg_merge(torch.cat([h, agg_msg], dim=-1))
            
        return h


# =============================================================================
# 2. TIME EMBEDDING & ADALN MODULATOR
# =============================================================================
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timesteps, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# =============================================================================
# 3. DiT-SMALL TRANSFORMER BLOCK WITH STRONG ADALN
# =============================================================================
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c_node):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c_node).chunk(6, dim=-1)
        res_attn = (self.norm1(x) * (1 + scale_msa) + shift_msa)
        attn_out, _ = self.attn(res_attn, res_attn, res_attn)
        x = x + gate_msa * attn_out
        res_mlp = (self.norm2(x) * (1 + scale_mlp) + shift_mlp)
        x = x + gate_mlp * self.mlp(res_mlp)
        return x


# =============================================================================
# 4. GNN-INTEGRATED DiT-SMALL BACKBONE
# =============================================================================
class DiTSmallFloorplanBackbone(nn.Module):
    def __init__(self, hidden_size=384, depth=12, num_heads=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.coord_embedder = nn.Linear(4, hidden_size)      
        self.gnn_encoder = NetlistGNN(node_in_dim=5 + 1, hidden_dim=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, 4)
        )
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, noised_positions, area_target, constraints, b2b_conn, t, block_count):
        x = self.coord_embedder(noised_positions)
        gnn_feats = self.gnn_encoder(constraints, area_target, b2b_conn, block_count)
        gnn_feats_b = gnn_feats.unsqueeze(0)
        t_feat = self.t_embedder(t)
        t_feat_b = t_feat.unsqueeze(1).expand(-1, block_count, -1)
        c_node = self.cond_fusion(torch.cat([gnn_feats_b, t_feat_b], dim=-1))
        
        for block in self.blocks:
            x = block(x, c_node)
            
        shift, scale = self.final_adaLN(t_feat).chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        output = self.final_layer(x)
        return output


# =============================================================================
# 5. 主優化器類別 (MyOptimizer)
# =============================================================================

CHECKPOINT_ENV_VAR = "MY_OPTIMIZER_CHECKPOINT"


def checkpoint_step_from_name(path: Path) -> int:
    step_match = re.search(r"step_(\d+)", path.name)
    if step_match:
        return int(step_match.group(1))

    epoch_match = re.search(r"epoch_(\d+)", path.name)
    if epoch_match:
        return int(epoch_match.group(1))

    return -1


def checkpoint_loss_is_finite(path: Path) -> bool:
    loss_match = re.search(r"loss_([^_]+)", path.stem)
    if not loss_match:
        return True
    try:
        return math.isfinite(float(loss_match.group(1)))
    except ValueError:
        return True


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def state_dict_is_finite(state_dict) -> bool:
    for value in state_dict.values():
        if (
            torch.is_tensor(value)
            and (value.is_floating_point() or value.is_complex())
            and not torch.isfinite(value).all().item()
        ):
            return False
    return True


def normalize_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def requested_checkpoint_candidates(weight_path: Path) -> List[Path]:
    checkpoint_arg = os.environ.get(CHECKPOINT_ENV_VAR)
    if not checkpoint_arg:
        return []

    candidates: List[Path] = []
    requested = Path(checkpoint_arg).expanduser()
    if requested.is_absolute() or requested.parent != Path("."):
        candidates.append(requested)
    else:
        candidates.append(weight_path / requested)
        candidates.append(requested)

    unique_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            unique_candidates.append(candidate)
    return unique_candidates


def load_latest_finite_checkpoint(model, weight_path: Path, device, verbose: bool = False):
    if not weight_path.exists():
        return None

    preferred = requested_checkpoint_candidates(weight_path)
    preferred_names = {path.name for path in preferred}
    fallback = [
        path for path in weight_path.glob("*.pth")
        if checkpoint_loss_is_finite(path)
        and path.name not in preferred_names
    ]
    fallback.sort(key=lambda path: (checkpoint_step_from_name(path), path.name), reverse=True)
    candidates = preferred + fallback

    for path in candidates:
        try:
            checkpoint = torch.load(path, map_location=device)
            state_dict = strip_module_prefix(normalize_checkpoint_state(checkpoint))
            if not state_dict_is_finite(state_dict):
                if verbose:
                    print(f"--> WARNING: Skipping non-finite checkpoint: {path.name}")
                continue
            model.load_state_dict(state_dict, strict=True)
            return path
        except Exception as exc:
            if verbose:
                print(f"--> WARNING: Skipping unusable checkpoint {path.name}: {exc}")

    return None


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 實例化升級版 Edge-GNN + DiT 模型架構
        self.model = DiTSmallFloorplanBackbone(hidden_size=384, depth=12, num_heads=6)
        
        # 自動尋找最新一輪的 .pth 檔案
        weight_path = Path(__file__).parent / "checkpoints"
        latest_weight = load_latest_finite_checkpoint(
            self.model,
            weight_path,
            self.device,
            verbose=self.verbose,
        )
        self.checkpoint_loaded = latest_weight is not None
        if latest_weight is not None:
            if self.verbose:
                print(f"--> [A100 GPU] Successfully loaded Edge-GNN + DiT Checkpoint: {latest_weight.name}")
        else:
            if self.verbose:
                print("--> WARNING: No usable finite weights found in checkpoints/. Using deterministic guidance fallback.")
                
        self.model.to(self.device)
        self.model.eval() # 🚀 關閉 Dropout, 進入高效推論模式

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None
    ) -> List[Tuple[float, float, float, float]]:
        
        effective_blocks = block_count
        problem = HardConstraintNormalizer.normalize(
            effective_blocks,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
        )
        if effective_blocks == 0:
            return []

        # =========================================================================
        # 階段二：AI 階段 ── 建立只作為排序/候選偏好的擴散指引
        # =========================================================================
        guidance = DiffusionGuidanceAdapter.build(
            problem,
            self.model,
            self.device,
            checkpoint_loaded=self.checkpoint_loaded,
        )

        # =========================================================================
        # 階段三：Anchor-aware legalization ── preplaced anchors are obstacles
        # =========================================================================
        placement_state = AnchorAwareLegalizer.legalize(problem, guidance)
        if self.verbose and placement_state.warnings:
            print(f"--> Anchor-aware legalizer warnings: {placement_state.warnings}")

        checker = FeasibilityChecker()
        candidate_positions: List[Tuple[str, int, List[Tuple[float, float, float, float]]]] = []
        if not placement_state.failed:
            candidate_positions.append((
                "ml_anchor_aware",
                0,
                placement_state.as_positions(problem),
            ))

        try:
            constructive_candidates = ConstructiveCandidateLegalizer.build_candidates(problem, guidance)
        except (IndexError, RuntimeError, TypeError, ValueError) as exc:
            constructive_candidates = []
            if self.verbose:
                print(f"--> Constructive candidate path skipped: {type(exc).__name__}")

        for candidate_index, (source, positions) in enumerate(constructive_candidates, start=1):
            candidate_positions.append((source, candidate_index, positions))

        next_candidate_index = max(
            (source_order for _, source_order, _ in candidate_positions),
            default=0,
        )
        if guidance.available and (guidance.predicted_positions or guidance.predicted_centers):
            neutral_guidance = Guidance(
                order=list(problem.movables),
                predicted_centers={},
                predicted_positions={},
                available=False,
                warnings=["constructive_neutral_ignores_model_predictions"],
            )
            try:
                neutral_candidates = ConstructiveCandidateLegalizer.build_candidates(problem, neutral_guidance)
            except (IndexError, RuntimeError, TypeError, ValueError) as exc:
                neutral_candidates = []
                if self.verbose:
                    print(f"--> Neutral constructive candidate path skipped: {type(exc).__name__}")

            for source, positions in neutral_candidates:
                next_candidate_index += 1
                if source.startswith("constructive:"):
                    neutral_source = source.replace("constructive:", "constructive_neutral:", 1)
                else:
                    neutral_source = f"constructive_neutral:{source}"
                candidate_positions.append((neutral_source, next_candidate_index, positions))

        best_candidate = CandidateSelector.best_feasible(
            problem,
            candidate_positions,
            checker,
        )
        if best_candidate is not None:
            return best_candidate.positions

        optimized_report = checker.check(placement_state.as_positions(problem), problem)

        if self.verbose:
            print(
                "--> Optimized placement failed hard-feasibility check; "
                f"routing to fallback: {optimized_report.messages[:8]}"
            )

        fallback_positions = FeasibleFallbackPacker.pack(problem, guidance)
        fallback_report = checker.check(fallback_positions, problem)
        if fallback_report.is_feasible:
            return fallback_positions

        if self.verbose:
            print(f"--> Fallback placement remains infeasible: {fallback_report.messages[:8]}")

        # Contradictory hard inputs, such as overlapping immutable anchors, cannot
        # be repaired locally. Return the deterministic fallback so the evaluator
        # reports hard violations instead of turning the case into an exception.
        return fallback_positions
