# SAMVG algorithm reference

This is a pseudocode reference for the two-phase method described in Chapter 3
of Yiding Zhu's *SAMVG* dissertation.  It is a behavioural specification for
Vectrify's SAMVG-inspired path, not a copy of unreleased research code.

The ordering below matters.  In particular, impact is scored for a complete
cleaned SAM mask before that mask is split into connected components for path
tracing.  Coverage prompts and residual prompts solve different problems and
must not be conflated.

## Parameters defined by the dissertation

```text
AUTOMATIC_POINT_GRID = 32 x 32
RESIDUAL_THRESHOLD  = 0.784
FIT_STEPS_PER_PHASE = 500
```

The dissertation leaves the SAM checkpoint, SAM confidence/stability gates,
small-region and hole thresholds, impact threshold, circular-kernel radius,
and optimiser hyperparameters as implementation choices.  Keep those as
explicit parameters and benchmark them; do not infer a canonical value from a
path-count target alone.

## Reported representation variations

The baseline uses a fixed number of cubic segments per closed contour and
opaque fills.  The dissertation also reports two independent variations:

```text
SAMVG+var   = select locally distinct contour points whose curvature score
              crosses a caller-selected threshold, then fit one cubic between
              each adjacent selected pair
SAMVG+alpha = make every path fill opacity an optimisation parameter
```

These are representation changes, not mask-selection changes.  They must be
enabled explicitly when comparing with an SVG made by either variation; the
threshold value itself is not specified by the dissertation.

## Data types

```text
Mask        = boolean H x W image
PaintedMask = (mask: Mask, colour: RGB)
Path        = closed filled SVG path
Document    = ordered list of SVG elements
Canvas      = RGB H x W image
```

`Composite(canvas, mask, colour)` paints `colour` wherever `mask` is true.
`Error(target, canvas)` is the pixel reconstruction error used consistently
within a filtering pass.  For a blank initial canvas, uncovered pixels receive
the maximum error so the first mask is not biased toward bright colours.

## Common mask preparation and impact filter

```text
function CLEAN_MASKS(raw_masks):
    # This is SAM automatic-mask post-processing, before SAMVG selection.
    masks = retain masks passing SAM's predicted-quality/stability gates
    masks = remove configured small connected regions and small holes
    return masks


function FILTER_BY_IMPACT(target, masks, initial_canvas, min_improvement):
    # Each entry remains a WHOLE cleaned SAM mask until it is accepted.
    candidates = CLEAN_MASKS(masks)
    candidates = sort candidates by descending mask area

    canvas = copy(initial_canvas)
    accepted = []

    for mask in candidates:
        colour = mean_rgb(target pixels where mask is true)
        proposal = Composite(canvas, mask, colour)

        improvement = Error(target, canvas) - Error(target, proposal)
        if improvement < min_improvement:
            continue

        accepted.append((mask, colour, improvement))
        canvas = proposal

    return accepted, canvas
```

Do **not** score disconnected components of one SAM mask independently.  Split
an accepted mask only when tracing it: every connected component becomes its
own editable SVG path, inherits the accepted mask colour, and preserves the
accepted mask's painter-order slot.

```text
function TRACE_ACCEPTED_MASKS(accepted_masks):
    document = []
    for (mask, colour, _) in accepted_masks in acceptance order:
        for component in connected_components(mask):
            contour = extract_outer_contour(component)
            path = fit_fixed_segment_bezier_path(contour)
            document.append(filled_path(path, colour))
    return document
```

## Phase 1: segmentation, coverage recovery, and first fit

```text
function FIRST_PHASE(target):
    raw = SAM_AUTOMATIC_MASK_GENERATION(
        target,
        point_grid = AUTOMATIC_POINT_GRID,
        crop_schedule = SAM_AMG_CROP_SCHEDULE,
    )
    # AMG removes duplicate candidates in two passes: predicted-IoU NMS
    # within each crop, then crop-area-priority NMS across all crop outputs.
    # The latter prefers a duplicate from the smaller crop.

    # Begin on a blank canvas and retain useful whole masks in area order.
    first_masks, mask_canvas = FILTER_BY_IMPACT(
        target, raw, blank_canvas(target.size), min_improvement
    )

    # This recovery looks for regions with no retained-mask coverage.  It is
    # still part of segmentation, before any SVG path optimisation.
    uncovered = NOT union(mask for (mask, _, _) in first_masks)
    coverage_map = circular_convolution(uncovered)
    coverage_centres = component_centres(threshold(coverage_map))
    coverage_raw = SAM_PROMPTED_MASKS(target, coverage_centres)

    # Score newly prompted masks against the retained-mask composite, not a
    # fresh blank canvas, and append accepted masks after existing masks.
    coverage_masks, _ = FILTER_BY_IMPACT(
        target, coverage_raw, mask_canvas, min_improvement
    )
    seed = TRACE_ACCEPTED_MASKS(first_masks + coverage_masks)

    first_fit = OPTIMISE_PATHS(seed, target, steps = FIT_STEPS_PER_PHASE)
    return seed, first_fit
```

The coverage pass finds *uncovered* areas.  It cannot recover texture inside a
large filled mask that already covers the relevant pixels.

## Phase 2: residual-detail recovery and second fit

```text
function SECOND_PHASE(target, first_fit):
    fitted_canvas = RASTERISE(first_fit)
    difference = sum_over_rgb(abs(target - fitted_canvas))

    # Unlike coverage recovery, this detects high-error regions after fitting,
    # including regions that are already alpha-covered by a coarse fill.
    residual_map = circular_convolution(difference)
    residual_regions = connected_components(
        threshold(residual_map, RESIDUAL_THRESHOLD)
    )
    residual_centres = centres(residual_regions)
    residual_raw = SAM_PROMPTED_MASKS(target, residual_centres)

    # Filter against the fitted render so only new masks that reduce remaining
    # error are kept.  Append their paths in painter order after first_fit.
    residual_masks, _ = FILTER_BY_IMPACT(
        target, residual_raw, fitted_canvas, min_improvement
    )
    additions = TRACE_ACCEPTED_MASKS(residual_masks)

    recovered_document = append_in_painter_order(first_fit, additions)
    final_fit = OPTIMISE_PATHS(
        recovered_document, target, steps = FIT_STEPS_PER_PHASE
    )
    return recovered_document, final_fit
```

## End-to-end procedure

```text
function SAMVG(target):
    seed, first_fit = FIRST_PHASE(target)
    recovered_document, final_fit = SECOND_PHASE(target, first_fit)
    return {
        seed,
        first_fit,
        recovered_document,
        final_fit,
    }
```

## Implementation invariants

- Preserve document/painter order throughout; accepted additions come after
  the canvas against which their impact was measured.
- Use the same cleanup, complete-mask impact filtering, and fixed-segment
  tracing procedure for automatic, coverage-prompted, and residual-prompted
  masks.
- Keep coverage recovery and residual recovery as separate passes.  A high
  coverage percentage does not show that residual detail recovery is needless.
- Judge a run by raster error and visual output, not just retained-mask or
  emitted-path count.  Component splitting can make these counts diverge.
- Optional stroke/text handling is an extension around this fill pipeline; it
  must not alter the fill-mask ordering or residual-recovery criteria.
