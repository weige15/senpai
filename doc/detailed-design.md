# Detailed Design

## Purpose

This document turns the proposal, high-level design, and test plan into an implementation-ready design for the hard-constraint-safe legalization path in `my_optimizer.py`. It defines private in-file module contracts, shared data shapes, algorithms, failure handling, and test mapping without writing production code.

The design goal is not to maximize contest score immediately. The first implementation target is to stop returning hard-infeasible layouts after diffusion guidance, because any overlap, soft-block area violation, fixed-shape dimension drift, or preplaced position/dimension drift receives infeasible cost `10.0`.

## Source Proposal Summary

The proposal identifies the current failure mode as an ordering problem in `my_optimizer.py`:

- dimensions are mostly normalized before packing;
- the Edge-GNN plus DiT model predicts an initial layout;
- `BStarTreeLegalizer` packs blocks using diffusion-derived order;
- a final correction step overwrites preplaced coordinates and moves boundary-constrained blocks to bounding-box edges.

The final overwrite/move step can break the non-overlap guarantee produced by the packer. The proposed fix is a feasibility-first legalizer that inserts preplaced blocks as immutable anchors before placing movable blocks, treats diffusion output only as advisory guidance, accepts soft-constraint moves only when they preserve hard feasibility, and falls back to a conservative packer when optimized legalization fails final checks.

The user clarified and approved that `data_lite/worker_*` exists only on the lab server as the training dataset, while `LiteTensorDataTest/` is the public local validation/evaluation set of 100 cases. This detailed design treats those roles as fixed.

## HLD Summary

The HLD defines a single in-process pipeline inside `my_optimizer.py`:

1. `HardConstraintNormalizer`
2. `DiffusionGuidanceAdapter`
3. `AnchorAwareLegalizer`
4. `SoftConstraintImprover`
5. `FeasibilityChecker`
6. `FeasibleFallbackPacker`

The HLD boundaries are preserved. These modules may be implemented as private classes, dataclasses, or helper functions inside `my_optimizer.py`; no new source files are required by this design.

Data ownership follows the HLD:

- `HardConstraintNormalizer` owns legal dimensions and immutability metadata.
- `DiffusionGuidanceAdapter` owns advisory order and predicted preferences, not legality.
- `AnchorAwareLegalizer` owns optimized hard-aware coordinates.
- `SoftConstraintImprover` owns optional legality-preserving soft moves.
- `FeasibilityChecker` owns evaluator-aligned hard-feasibility results.
- `FeasibleFallbackPacker` owns conservative recovery if the optimized path fails.

## Design Goals

- Preserve the `FloorplanOptimizer.solve(...)` external API and return one `(x, y, width, height)` tuple per block.
- Keep implementation scoped to `my_optimizer.py` unless later approved.
- Preserve exact dimensions for fixed-shape and preplaced blocks.
- Preserve exact coordinates and dimensions for preplaced blocks.
- Keep soft-block realized area within 1% relative error.
- Prevent positive-area rectangle overlap; touching edges is allowed.
- Treat boundary, grouping, and MIB as soft score constraints only.
- Use diffusion output for ordering and candidate preference only.
- Make hard-feasibility checks evaluator-aligned using approved tolerances:
  - overlap threshold: positive overlap on both axes greater than `1e-6`;
  - fixed/preplaced position and dimension tolerance: `1e-4`;
  - soft-block area tolerance: `1%`.
- Keep legalizer and fallback deterministic for reproducible failures.
- Keep placement practical for 21-120 block public validation cases.

## Non-Goals

- Do not edit `iccad2026_evaluate.py`, validation data, training data, labels, or `.pth` dataset files.
- Do not require new dependencies or a new package layout.
- Do not add training as a minimum correctness gate.
- Do not split `data_lite` into validation data for this design.
- Do not treat `data_lite/worker_*` as locally available in this workspace.
- Do not treat `LiteTensorDataTest/` as training data.
- Do not make boundary, grouping, or MIB constraints hard feasibility constraints.
- Do not solve final hidden-test packaging or official runtime normalization.
- Do not define a custom behavior for contradictory hard inputs where both optimized placement and fallback cannot be made feasible; that remains an open question.

## Architecture Overview

The design is a private helper pipeline called from `MyOptimizer.solve(...)`.

```text
solve inputs
  -> HardConstraintNormalizer.normalize()
  -> DiffusionGuidanceAdapter.build()
  -> AnchorAwareLegalizer.legalize()
  -> SoftConstraintImprover.improve()
  -> FeasibilityChecker.check()
      -> return if feasible
      -> FeasibleFallbackPacker.pack()
      -> FeasibilityChecker.check()
      -> return if feasible
      -> unresolved hard-failure behavior
```

Key rules:

- Raw `target_positions` sentinel values are normalized once; later modules should not reinterpret `-1` fields independently.
- Legal dimensions are authoritative after normalization.
- Preplaced anchors are occupied rectangles before movable placement begins.
- Movable block coordinates are chosen only from hard-feasible candidates.
- Soft moves are optional and can be a no-op in the first implementation.
- Final output order is block-index order, not placement order.

## Shared Data Contracts

These are private in-file contracts for `my_optimizer.py`. Names may be implemented as dataclasses or equivalent simple records, but the fields and invariants are the design contract.

`PlacementTuple`

- Shape: `(x: float, y: float, width: float, height: float)`.
- Invariants: all fields finite; `width > 0`; `height > 0`.
- Semantics: lower-left coordinate plus dimensions.

`BlockSpec`

- `index: int`
- `area_target: float`
- `width: float`
- `height: float`
- `is_fixed: bool`
- `is_preplaced: bool`
- `preplaced_x: Optional[float]`
- `preplaced_y: Optional[float]`
- `boundary_mask: int`
- `mib_group: Optional[int]`
- `cluster_group: Optional[int]`

Invariants:

- Fixed-shape and preplaced dimensions come from `target_positions[i, 2:4]`.
- Preplaced blocks have `preplaced_x` and `preplaced_y`.
- Soft blocks have area-preserving dimensions derived from `area_targets`.
- Raw `-1` sentinels from `target_positions` do not escape normalization except as `None`.

`NormalizedProblem`

- `block_count: int`
- `blocks: list[BlockSpec]`, length `block_count`
- `anchors: list[int]`, block indices where `is_preplaced`
- `movables: list[int]`, block indices where not preplaced
- `constraints: torch.Tensor`
- `area_targets: torch.Tensor`
- `target_positions: Optional[torch.Tensor]`
- `b2b_connectivity: torch.Tensor`
- `p2b_connectivity: torch.Tensor`
- `pins_pos: torch.Tensor`

Invariants:

- `anchors` and `movables` are disjoint and cover all block indices.
- The block list is indexed by block id.
- Connectivity tensors remain read-only advisory inputs.

`Guidance`

- `order: list[int]` for movable block placement.
- `predicted_centers: dict[int, tuple[float, float]]`.
- `predicted_positions: dict[int, tuple[float, float]]`.
- `available: bool`.
- `warnings: list[str]`.

Invariants:

- Contains no legal dimensions.
- Excludes anchors from placement order unless a later diagnostic needs them.
- Falls back deterministically if model output is missing, invalid, or non-finite.

`PlacementState`

- `positions: list[Optional[PlacementTuple]]`, length `block_count`.
- `occupied: list[tuple[int, PlacementTuple]]`.
- `placed_order: list[int]`.

Invariants:

- Anchors are inserted with exact target coordinates and dimensions.
- Every occupied rectangle corresponds to exactly one placed block.
- A block is added to `occupied` only after passing candidate overlap checks.

`FeasibilityReport`

- `is_feasible: bool`
- `overlap_violations: int`
- `area_violations: int`
- `dimension_violations: int`
- `malformed_violations: int`
- `messages: list[str]`

Invariants:

- `is_feasible` is true only when all violation counts are zero.
- Counts are diagnostic and used for fallback routing.

## Module Designs

### HardConstraintNormalizer

#### Responsibility

Convert raw evaluator inputs into legal per-block metadata before model output or placement decisions are interpreted. This module identifies preplaced, fixed-shape, and movable soft blocks; derives authoritative dimensions; and carries soft-constraint annotations forward.

#### Non-Responsibility

- Does not run the neural model.
- Does not choose coordinates for movable blocks.
- Does not compute HPWL, area score, or soft-constraint score.
- Does not repair contradictory hard inputs such as overlapping preplaced anchors.

#### Inputs and Outputs

Inputs:

- `block_count`
- `area_targets`
- `b2b_connectivity`
- `p2b_connectivity`
- `pins_pos`
- `constraints`
- `target_positions`

Output:

- `NormalizedProblem`

#### Public Interface

Private in-file interface:

```python
normalize(
    block_count,
    area_targets,
    b2b_connectivity,
    p2b_connectivity,
    pins_pos,
    constraints,
    target_positions,
) -> NormalizedProblem
```

This is not an external API. It is called only by `MyOptimizer.solve(...)` or an equivalent private orchestrator.

#### Data Structures

Uses `BlockSpec` and `NormalizedProblem`.

Constraint interpretation:

- `constraints[:, 0] != 0`: fixed-shape block.
- `constraints[:, 1] != 0`: preplaced block.
- `constraints[:, 2]`: MIB annotation, passed through as soft metadata.
- `constraints[:, 3]`: grouping/cluster annotation, passed through as soft metadata.
- `constraints[:, 4]`: boundary bitmask, with left `1`, right `2`, top `4`, bottom `8`.

Target-position interpretation:

- Free soft block: all target fields may be `-1`.
- Fixed-shape block: width and height are expected in `target_positions[i, 2:4]`.
- Preplaced block: x, y, width, and height are expected in `target_positions[i, 0:4]`.

#### Internal Design

For each block index `i` from `0` to `block_count - 1`:

1. Read fixed/preplaced flags from `constraints` when the tensor has the expected columns.
2. If fixed or preplaced and target dimensions are present, set `width` and `height` exactly from `target_positions`.
3. Otherwise derive soft dimensions from the area target. The first implementation may use square dimensions, `sqrt(area) x sqrt(area)`, because aspect ratio is relaxed for soft blocks.
4. If preplaced, set immutable `preplaced_x` and `preplaced_y` exactly from `target_positions`.
5. Record soft annotations from MIB, cluster, and boundary columns without enforcing them as hard constraints.

#### Algorithm Details

Pseudocode:

```text
blocks = []
anchors = []
movables = []
for i in range(block_count):
    flags = read constraint row i
    is_fixed = constraints[i, 0] != 0 if present else false
    is_preplaced = constraints[i, 1] != 0 if present else false

    if target_positions has valid width and height for fixed/preplaced:
        width = target_positions[i, 2]
        height = target_positions[i, 3]
    else:
        area = max(float(area_targets[i]), 1.0)
        width = sqrt(area)
        height = sqrt(area)

    if is_preplaced:
        preplaced_x = target_positions[i, 0]
        preplaced_y = target_positions[i, 1]
        anchors.append(i)
    else:
        movables.append(i)

    blocks.append(BlockSpec(...))
```

Complexity: `O(n)` for `n = block_count`, excluding tensor conversion overhead.

#### Dependencies

- Uses only raw solve inputs.
- Depends on evaluator constraint semantics documented in `iccad2026_evaluate.py` and the planning docs.
- Does not depend on diffusion, legalizer, checker, or fallback modules.

#### Failure Handling

- Missing `target_positions` for fixed or preplaced blocks is recorded as malformed metadata for later `FeasibilityChecker` rejection.
- Non-positive soft area targets are normalized to a positive safe dimension only if necessary to produce finite dimensions; the original target remains available for checker diagnostics.
- Contradictory hard inputs are not solved here.

#### Independent Test Plan

- Soft-block dimension derivation preserves `width * height` within 1%.
- Fixed-shape dimensions exactly match `target_positions`.
- Preplaced coordinates and dimensions exactly match `target_positions`.
- Boundary bitmask is preserved as soft metadata.
- Missing or malformed target fields are surfaced for checker failure, not silently treated as legal anchors.

#### Open Questions

- Should malformed fixed/preplaced metadata be logged immediately during normalization or only reported by `FeasibilityChecker`?

### DiffusionGuidanceAdapter

#### Responsibility

Convert existing model output into advisory placement order and candidate preferences. This module keeps the trained or initialized model useful for quality while preventing model predictions from changing legal dimensions, anchors, or hard feasibility.

#### Non-Responsibility

- Does not decide final coordinates.
- Does not resize blocks.
- Does not move preplaced anchors.
- Does not validate hard constraints.
- Does not require checkpoints to exist.

#### Inputs and Outputs

Inputs:

- `NormalizedProblem`
- existing model instance from `MyOptimizer`
- model device

Output:

- `Guidance`

#### Public Interface

Private in-file interface:

```python
build(problem: NormalizedProblem, model, device) -> Guidance
```

#### Data Structures

Uses `Guidance`, legal dimensions from `BlockSpec`, and the existing tensor inputs already accepted by the current model path.

#### Internal Design

The adapter uses normalized dimensions to construct the model input but only consumes predicted x/y or center-like values as advisory signals. Predicted width and height values are ignored for legality because `HardConstraintNormalizer` owns dimensions.

If the model output is unavailable, raises an exception, has an unexpected shape, or contains non-finite values, the adapter returns deterministic fallback guidance:

- `available = False`
- `order = problem.movables` in stable block-index order or another deterministic order explicitly chosen in implementation
- empty predicted preferences
- warning message for diagnostics when verbose mode is enabled

#### Algorithm Details

Pseudocode:

```text
try:
    model_input = build tensors from normalized legal dimensions
    predicted = model(...)
    for movable block i:
        read predicted x/y
        if finite:
            predicted_positions[i] = (x, y)
            predicted_centers[i] = (x + width/2, y + height/2)
    order = stable sort of movables by predicted x, predicted y, then block index
    if prediction coverage is incomplete:
        fill missing movables with deterministic order
    return Guidance(order, predicted data, available=true)
except expected runtime/model errors:
    return fallback Guidance
```

The exact fallback ordering policy is an implementation detail, but it must be deterministic and must not affect hard feasibility.

#### Dependencies

- Depends on `HardConstraintNormalizer` output.
- Depends on existing model components already in `my_optimizer.py`.
- Does not depend on the legalizer, checker, fallback, or training dataset.

#### Failure Handling

- Missing `checkpoints/` is not a failure.
- Invalid model output degrades to deterministic fallback guidance.
- CUDA memory cleanup may still happen after model inference if the implementation already does so, but guidance failure must not prevent fallback placement.

#### Independent Test Plan

- With synthetic finite predictions, returns a stable order and predicted centers.
- With non-finite predictions, returns deterministic fallback guidance.
- With missing checkpoints, optimizer remains able to produce guidance fallback.
- Predicted dimensions never modify `BlockSpec.width` or `BlockSpec.height`.
- Preplaced anchors are excluded from movable placement order.

#### Open Questions

- Should fallback order be block-index order or descending block area with block-index tie-break? The design requires determinism either way.

### AnchorAwareLegalizer

#### Responsibility

Produce a non-overlapping optimized placement by packing movable blocks around preplaced anchors. Diffusion guidance may influence order and candidate scoring, but every accepted candidate must satisfy hard feasibility against already occupied rectangles.

#### Non-Responsibility

- Does not alter legal dimensions.
- Does not move anchors.
- Does not make soft constraints hard.
- Does not provide the final authority on hard feasibility; `FeasibilityChecker` remains the final checker.
- Does not handle unrecoverable contradictory anchors beyond returning a failed candidate or diagnostic.

#### Inputs and Outputs

Inputs:

- `NormalizedProblem`
- `Guidance`

Output:

- `PlacementState` or list of `PlacementTuple` candidates in block-index order.

#### Public Interface

Private in-file interface:

```python
legalize(problem: NormalizedProblem, guidance: Guidance) -> PlacementState
```

#### Data Structures

Uses:

- `BlockSpec` for legal dimensions.
- `PlacementState` for positions and occupied rectangles.
- candidate records with `(x, y, score, reason)` as private local values.

Candidate scoring may use:

- hard-feasibility filter;
- distance to predicted center when guidance exists;
- bounding-box area growth;
- local HPWL contribution against already placed connected blocks and pins;
- soft boundary preference only as a score term, never as a hard filter.

#### Internal Design

Initialization:

1. Create empty `PlacementState`.
2. Insert each preplaced anchor at exact `(preplaced_x, preplaced_y, width, height)`.
3. Add anchor rectangles to `occupied`.
4. If anchors overlap, leave detection to immediate feasibility checking or return a state that will be rejected.

Movable placement:

1. Iterate through `guidance.order`.
2. Build a bounded deterministic candidate set.
3. Reject candidates that would overlap any occupied rectangle.
4. Score remaining candidates.
5. Choose the lowest-score candidate with deterministic tie-breakers.
6. Add the chosen rectangle to state.

The candidate-set implementation should be obstacle-aware. A practical first implementation can generate candidates from:

- origin or current bounding-box corner candidates;
- right-edge and top-edge points of occupied rectangles;
- candidate points near diffusion-predicted x/y projections;
- conservative strip/shelf fallback points if no preferred candidate is valid.

Candidate generation must be bounded so runtime remains practical for 21-120 blocks.

#### Algorithm Details

Pseudocode:

```text
state = empty PlacementState
for anchor in problem.anchors:
    place exact anchor tuple
    occupied.append(anchor rect)

for block_id in guidance.order:
    block = problem.blocks[block_id]
    candidates = generate_bounded_candidates(block, state, guidance)
    feasible = []
    for c in candidates sorted deterministically:
        rect = (c.x, c.y, block.width, block.height)
        if rect is finite and does_not_overlap(rect, state.occupied):
            feasible.append((score_candidate(rect), rect))

    if feasible is empty:
        rect = find_conservative_nonoverlap_point(block, state)
    else:
        rect = lowest score with deterministic tie-break

    if rect is found:
        state.positions[block_id] = rect
        state.occupied.append((block_id, rect))
    else:
        mark placement failure

return state
```

Complexity target: near-quadratic for public validation sizes. Candidate generation should avoid unbounded growth; if the first implementation uses an `O(n^3)` exhaustive candidate scan, it must be treated as a temporary implementation risk and benchmarked before full validation.

#### Dependencies

- Depends on `HardConstraintNormalizer` for dimensions and anchors.
- Depends on `DiffusionGuidanceAdapter` for advisory order and preferences.
- May use lightweight local overlap helpers aligned with `FeasibilityChecker`.
- Does not call evaluator CLI or read datasets.

#### Failure Handling

- If no optimized candidate is found for a movable block, the module returns a failed/incomplete state that triggers fallback.
- If anchors overlap, final feasibility rejects the state.
- If guidance is missing, uses deterministic placement order.
- If a scoring term cannot be computed because a connected peer is not yet placed, that term is skipped for that candidate.

#### Independent Test Plan

- One preplaced anchor plus one movable block: movable avoids anchor.
- Multiple anchors with gaps: movable blocks either fill legal spaces or expand the layout.
- Touching rectangles are accepted.
- Positive-area intersections are rejected.
- Missing guidance still produces deterministic placement.
- Candidate tie-breaking is reproducible for identical inputs.

#### Open Questions

- What exact candidate cap should be used to balance quality and runtime?
- Should HPWL contribution be part of the first implementation, or deferred until hard feasibility is stable?

### SoftConstraintImprover

#### Responsibility

Optionally improve boundary, grouping, and MIB soft constraints through transformations that preserve hard feasibility. This module can be a no-op in the first implementation if hard feasibility is not yet stable.

#### Non-Responsibility

- Does not repair hard-constraint violations from the legalizer.
- Does not move preplaced anchors.
- Does not resize fixed-shape or preplaced blocks.
- Does not accept a move without hard-feasibility checking.

#### Inputs and Outputs

Inputs:

- `NormalizedProblem`
- legalizer placement state or placement list
- `FeasibilityChecker`

Output:

- same placement or improved placement

#### Public Interface

Private in-file interface:

```python
improve(problem: NormalizedProblem, placement, checker: FeasibilityChecker) -> placement
```

#### Data Structures

Uses `PlacementTuple` list and soft annotations stored in `BlockSpec`.

Candidate move examples:

- boundary translation for blocks with nonzero boundary mask;
- MIB shape synchronization for soft blocks only, if implemented later;
- grouping compaction moves for blocks in the same cluster, if implemented later.

#### Internal Design

First implementation option:

- return the input placement unchanged.

Incremental boundary implementation:

1. Compute current bounding box.
2. For each non-preplaced block with a boundary mask, generate one or more target translations to touch requested edges.
3. Reject moves that change dimensions.
4. Reject moves that overlap another block.
5. Run `FeasibilityChecker` on the full candidate placement before accepting.

Grouping and MIB improvements should be deferred unless their exact constraint encoding and move rules are clear.

#### Algorithm Details

Pseudocode for boundary-only improvement:

```text
best = placement
for block with boundary mask:
    if block is preplaced:
        continue
    for target move implied by boundary mask:
        candidate = copy(best)
        candidate[block] = translated tuple with same width/height
        if checker.check(candidate).is_feasible:
            best = candidate
return best
```

Complexity: boundary-only improvement is `O(m * check_cost)` for `m` candidate moves. It should be skipped or capped if it becomes expensive.

#### Dependencies

- Depends on `FeasibilityChecker`.
- Depends on `HardConstraintNormalizer` soft annotations.
- Does not depend on diffusion or training data.

#### Failure Handling

- Failed or unsafe soft moves are rejected without modifying the current feasible placement.
- If the input placement is infeasible, this module should not attempt broad repair; fallback owns recovery.
- If soft constraint encoding is ambiguous, the module should no-op and record an open question rather than enforcing incorrect behavior.

#### Independent Test Plan

- Boundary move that would overlap is rejected.
- Boundary move that is safe is accepted only if all hard checks pass.
- Preplaced block with boundary mask is not moved.
- No-op implementation preserves input placement exactly.
- Grouping/MIB moves are marked missing if not implemented.

#### Open Questions

- What exact grouping and MIB move rules should be used once hard feasibility is stable?
- Should boundary moves be attempted before or after local compaction, if local compaction is later added?

### FeasibilityChecker

#### Responsibility

Provide evaluator-aligned hard-constraint checks before returning a solution or accepting a soft move. It reports overlap, area, fixed/preplaced, and malformed-output violations.

#### Non-Responsibility

- Does not compute contest score.
- Does not optimize placement quality.
- Does not decide fallback placement coordinates.
- Does not reinterpret raw constraints beyond normalized metadata and approved evaluator semantics.

#### Inputs and Outputs

Inputs:

- candidate placement list
- `NormalizedProblem`

Output:

- `FeasibilityReport`

#### Public Interface

Private in-file interface:

```python
check(positions: list[PlacementTuple], problem: NormalizedProblem) -> FeasibilityReport
```

Optional helper:

```python
rectangles_overlap(a: PlacementTuple, b: PlacementTuple) -> bool
```

#### Data Structures

Uses `FeasibilityReport`.

Violation categories:

- malformed count: wrong length, non-finite values, non-positive dimensions;
- overlap count: pairwise positive-area intersections beyond `1e-6`;
- area count: soft blocks with relative area error greater than `0.01`;
- dimension count: fixed/preplaced width/height mismatch beyond `1e-4`, and preplaced x/y mismatch beyond `1e-4`.

#### Internal Design

The checker mirrors the hard-feasibility semantics in `iccad2026_evaluate.py`:

- `check_overlap`: overlap exists only when both x-axis and y-axis intersection lengths are greater than `1e-6`.
- `check_area_tolerance`: fixed and preplaced blocks are excluded from soft area tolerance because exact dimensions are checked separately.
- `check_dimension_hard_constraints`: fixed dimensions and preplaced position/dimensions use `1e-4` tolerance.

The module may call evaluator helper functions for final full-placement checks if those imports remain available, but it should keep local helper logic available for candidate move checks and to avoid coupling soft-move acceptance to evaluator CLI behavior.

#### Algorithm Details

Pseudocode:

```text
report = zero counts
if len(positions) != problem.block_count:
    report.malformed_violations += 1

for each tuple:
    validate finite x/y/w/h and positive w/h

for each pair i < j:
    if overlap_x > 1e-6 and overlap_y > 1e-6:
        report.overlap_violations += 1

for each block:
    if fixed or preplaced:
        compare width/height to normalized target with 1e-4 tolerance
        if preplaced, compare x/y to anchor coordinate with 1e-4 tolerance
    else:
        compare width * height to area_target with 1% tolerance

report.is_feasible = all counts are zero
```

Complexity: pairwise overlap check is `O(n^2)`, acceptable for 21-120 block validation cases.

#### Dependencies

- Depends on `NormalizedProblem`.
- Mirrors evaluator helper semantics.
- Does not depend on diffusion, legalizer, fallback, or training data.

#### Failure Handling

- Malformed placements are infeasible.
- Missing required fixed/preplaced target metadata is infeasible.
- Contradictory hard inputs are reported through violation counts, but resolution remains outside this module.

#### Independent Test Plan

- Touching rectangles produce zero overlap violations.
- Positive-area intersections produce overlap violations.
- Soft area relative error exactly at or below 1% passes; above 1% fails.
- Fixed/preplaced dimensions are checked with `1e-4` tolerance.
- Preplaced x/y mismatch is checked with `1e-4` tolerance.
- Missing, extra, non-finite, or non-positive tuples are rejected.

#### Open Questions

- Should reports be printed in verbose mode, stored internally for later inspection, or both?

### FeasibleFallbackPacker

#### Responsibility

Produce a conservative hard-feasible placement when the optimized legalizer or soft improver fails final hard checks. It prioritizes feasibility over HPWL, bounding-box area, and soft-constraint score.

#### Non-Responsibility

- Does not train or run the model.
- Does not improve soft constraints except incidentally.
- Does not attempt expensive global optimization.
- Does not define final behavior for impossible contradictory hard inputs.

#### Inputs and Outputs

Inputs:

- `NormalizedProblem`
- optional `Guidance` for stable ordering if safe

Output:

- placement list in block-index order

#### Public Interface

Private in-file interface:

```python
pack(problem: NormalizedProblem, guidance: Optional[Guidance] = None) -> list[PlacementTuple]
```

#### Data Structures

Uses `BlockSpec`, `PlacementState`, and local row/strip state.

#### Internal Design

The fallback starts from anchors and places every movable block at a conservative non-overlapping location. A first implementation can use a row/strip strategy:

1. Insert exact anchors.
2. Choose deterministic movable order, optionally using `guidance.order` if available.
3. Maintain a cursor `(x, y)` and row height.
4. For each movable block, find the first cursor location that does not overlap anchors or previously placed blocks.
5. If a row location collides, advance x; if the row grows too wide, start a new row.
6. Expand the layout as needed rather than forcing blocks into the current bounding box.

The fallback must be accepted only after `FeasibilityChecker.check(...)` passes.

#### Algorithm Details

Pseudocode:

```text
state = anchors inserted exactly
cursor_x = min safe x
cursor_y = min safe y
row_height = 0
for block_id in deterministic_movable_order:
    while true:
        rect = (cursor_x, cursor_y, width, height)
        if does_not_overlap(rect, state.occupied):
            place rect
            cursor_x = rect.x + rect.width
            row_height = max(row_height, rect.height)
            break
        cursor_x = next_candidate_x_after_collision(rect, occupied)
        if cursor_x exceeds row_limit:
            cursor_x = row_start_x
            cursor_y = cursor_y + row_height
            row_height = 0
return positions
```

The exact row limit is an implementation choice, but it must be deterministic. When anchors make a row crowded, fallback should expand the layout rather than overlap an anchor.

Complexity target: `O(n^2)` pairwise collision checks for normal cases.

#### Dependencies

- Depends on `HardConstraintNormalizer` for dimensions and anchors.
- Depends on `FeasibilityChecker` for final acceptance, but should not call it repeatedly in a way that causes avoidable runtime growth.
- May consume `Guidance.order` only if doing so cannot compromise feasibility.

#### Failure Handling

- If anchors already overlap, fallback cannot guarantee feasibility; the final checker reports failure.
- If a block has invalid dimensions, fallback cannot place it legally; the final checker reports failure.
- If no safe finite position can be found due to malformed input, fallback returns a placement report or incomplete state for unresolved hard-failure behavior.

#### Independent Test Plan

- Soft-only blocks are packed without overlap.
- Preplaced anchors remain exact.
- Movable blocks avoid anchors.
- Fallback handles missing guidance.
- Fallback output passes `FeasibilityChecker` for normal synthetic cases.
- Overlapping immutable anchors are reported as infeasible rather than silently hidden.

#### Open Questions

- What should `MyOptimizer.solve(...)` return if fallback output fails final feasibility due to contradictory hard inputs?
- What deterministic row limit should be used for best balance of compactness and simplicity?

## Cross-Module Contracts

| Contract | Producer | Consumer | Rule |
| --- | --- | --- | --- |
| Legal dimensions | `HardConstraintNormalizer` | all later modules | Later modules must not alter fixed/preplaced dimensions or soft dimensions unless a future approved soft-block aspect strategy preserves area and passes checker. |
| Anchor immutability | `HardConstraintNormalizer` | legalizer, fallback, checker | Preplaced x/y/w/h are exact target values and cannot be moved by optimized, soft-improvement, or fallback paths. |
| Advisory guidance | `DiffusionGuidanceAdapter` | legalizer, fallback | Guidance may rank or score candidates but cannot make an infeasible candidate acceptable. |
| Occupied rectangles | legalizer/fallback | checker | Any returned placement must be block-index ordered and complete. |
| Soft moves | `SoftConstraintImprover` | checker | A soft move is accepted only if the full placement remains hard-feasible. |
| Final acceptance | `FeasibilityChecker` | `MyOptimizer.solve(...)` | Only a feasible optimized or fallback placement should be returned for normal non-contradictory inputs. |
| Dataset role | documentation and evaluator workflow | implementation planning | `LiteTensorDataTest/` is validation/evaluation; lab-server `data_lite/worker_*` is training-only and not used as validation by this design. |

## End-to-End Workflow

1. `MyOptimizer.solve(...)` receives evaluator inputs.
2. `HardConstraintNormalizer.normalize(...)` produces `NormalizedProblem`.
3. `DiffusionGuidanceAdapter.build(...)` attempts to produce advisory guidance from the existing model. Missing or invalid model output becomes deterministic fallback guidance.
4. `AnchorAwareLegalizer.legalize(...)` inserts anchors and places movable blocks around them.
5. `SoftConstraintImprover.improve(...)` optionally attempts checked soft moves. It may initially return the legalizer placement unchanged.
6. `FeasibilityChecker.check(...)` validates the optimized placement.
7. If feasible, `solve()` returns the block-index ordered placement.
8. If infeasible, `FeasibleFallbackPacker.pack(...)` produces conservative placement.
9. `FeasibilityChecker.check(...)` validates fallback.
10. If fallback is feasible, `solve()` returns fallback placement.
11. If fallback is infeasible, behavior remains an open question because contradictory hard inputs or malformed metadata need an explicit policy before implementation.

## Test Strategy Mapping

| Test-Plan Requirement | Design Coverage | Verification Level |
| --- | --- | --- |
| Return one tuple per block | `MyOptimizer.solve(...)` workflow and `FeasibilityChecker` malformed checks | smoke, integration |
| No overlaps | `AnchorAwareLegalizer`, `FeasibleFallbackPacker`, `FeasibilityChecker` | unit, integration, evaluator |
| Soft-block area within 1% | `HardConstraintNormalizer`, `FeasibilityChecker` | unit, integration |
| Fixed-shape dimensions unchanged | `HardConstraintNormalizer`, `FeasibilityChecker` | unit, integration |
| Preplaced position/dimensions unchanged | `HardConstraintNormalizer`, `AnchorAwareLegalizer`, `FeasibleFallbackPacker`, `FeasibilityChecker` | unit, integration |
| Boundary/grouping/MIB remain soft | `SoftConstraintImprover` no-op or checked moves only | unit, integration |
| Diffusion is advisory only | `DiffusionGuidanceAdapter` ignores predicted dimensions; legalizer filters candidates | unit |
| Preplaced anchors inserted first | `AnchorAwareLegalizer` and `FeasibleFallbackPacker` initialization | unit, integration |
| Evaluator-aligned final check | `FeasibilityChecker` tolerances and categories | unit, evaluator |
| Conservative fallback | `FeasibleFallbackPacker` and end-to-end fallback workflow | unit, integration |
| Deterministic behavior | stable orders, deterministic candidate tie-breaks, deterministic fallback | unit, regression |
| Public local validation set | quality gates run against `LiteTensorDataTest/` after approval | evaluator |
| Missing unit/lint/type gates recorded | quality gate section | documentation |
| Golden single-block and two-block cases | normalizer, checker, legalizer/fallback | unit |
| Boundary conflict case | `SoftConstraintImprover` checked move rejection | unit/integration |
| Randomized no-overlap and anchor immutability | legalizer, fallback, checker | property tests if harness is added |
| Large-case candidate explosion | bounded candidate generation and performance benchmark | benchmark |

## Quality Gates

No command was run while creating this document.

Documentation-only gate:

- Review `doc/detailed-design.md` and confirm it preserves HLD module names, reflects the approved dataset split, and does not change evaluator behavior.

Known, not run evaluator and grading commands:

- `python iccad2026_evaluate.py --validate my_optimizer.py`
- `python iccad2026_evaluate.py --validate my_optimizer.py --quick`
- `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`
- `python iccad2026_evaluate.py --evaluate my_optimizer.py`
- `python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions`
- `python iccad2026_evaluate.py --score my_optimizer_solutions.json`
- `python iccad2026_evaluate.py --baseline`
- `python iccad2026_evaluate.py --info`
- `python iccad2026_evaluate.py --visualize --test-id 0`

Implementation minimum gates after code changes, requiring approval before execution:

- `python iccad2026_evaluate.py --validate my_optimizer.py` passes.
- `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0` reports a feasible result.
- `python iccad2026_evaluate.py --evaluate my_optimizer.py` reports 100/100 feasible local public validation cases before score tuning is considered successful.

Missing gates:

- No unit-test command discovered.
- No lint command discovered.
- No format-check command discovered.
- No type-check command discovered.
- No CI configuration discovered.

Training is not a correctness gate:

- `python training_example.py` is long-running, writes checkpoints, and uses training data. It must be approved and scheduled separately.
- Lab-server `data_lite/worker_*` is training-only context, not local public validation.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Conservative fallback worsens HPWL and area | Accept as first-phase tradeoff; feasibility avoids cost `10.0`; improve quality after 100/100 feasible. |
| Candidate generation grows too large | Bound candidate set and benchmark on 21-120 block cases before full tuning. |
| Soft moves reintroduce hard violations | Start with no-op or boundary-only moves; require `FeasibilityChecker` acceptance for every move. |
| Missing checkpoints produce weak diffusion guidance | Guidance adapter degrades to deterministic order; legalizer and fallback own feasibility. |
| Partial workspace lacks helper modules | Keep commands as known/not-run until approved; report import failures as environment blockers, not design failures. |
| Fixed/preplaced target metadata malformed | Normalizer preserves diagnostics; checker marks output infeasible. |
| Contradictory preplaced anchors make feasibility impossible | Keep final behavior as open question; do not claim guaranteed feasibility for contradictory hard inputs. |
| Training/validation data confusion | Treat `data_lite/worker_*` as lab-server training-only and `LiteTensorDataTest/` as public local validation/evaluation. |

## Assumptions

- The implementation target is `my_optimizer.py`.
- The six HLD modules are implemented as private in-file helpers, classes, or dataclasses.
- No new source files are required by this detailed design.
- The external `FloorplanOptimizer.solve(...)` signature remains unchanged.
- `target_positions` follows evaluator semantics: fixed-shape blocks receive width/height; preplaced blocks receive x/y/width/height; free fields use `-1`.
- Approved hard-check tolerances are `1e-6` overlap axis threshold, `1e-4` fixed/preplaced tolerance, and `1%` soft area tolerance.
- `LiteTensorDataTest/` is the public local 100-case validation/evaluation set.
- Lab-server `data_lite/worker_*` is training-only and should not be used as validation unless a separate split policy is designed later.
- Missing checkpoints are allowed; optimizer behavior must remain hard-feasible without them for normal non-contradictory inputs.
- Boundary constraints remain soft even when encoded in `constraints[:, 4]`.
- Grouping and MIB soft-improvement details may be deferred until hard feasibility is stable.

## Open Questions

- What exact hard violation is most common in the current optimizer once an approved evaluator run is performed?
- Should `FeasibilityChecker` diagnostics be printed in verbose mode, stored in an internal field, or both?
- What should `MyOptimizer.solve(...)` return or raise if both optimized placement and fallback fail final hard-feasibility checks?
- Should `DiffusionGuidanceAdapter` fallback order use block-index order or descending area with block-index tie-break?
- What candidate cap should `AnchorAwareLegalizer` use to balance quality and runtime?
- Should HPWL-aware scoring be implemented in the first legalizer pass or deferred until after 100/100 feasibility?
- What deterministic row limit should `FeasibleFallbackPacker` use?
- Is adding a small internal test harness acceptable in the implementation phase, given no test framework exists today?
- Are all original FloorSet helper modules available in the intended evaluation environment?
