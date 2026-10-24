# Representative Model Comparison Design

## Goal

Fill UI cell 20 of `paper/a_paper_cvae_result.ipynb` with a representative-parameter comparison that shows Monte Carlo, CVAE, and CRealNVP together.

## Behavior

- Reuse the current notebook's representative parameters and the MC/CVAE result pickle files produced by the preceding cells.
- Select the matching BS or Heston CRealNVP checkpoint through an editable mapping at the top of the cell.
- Reuse checkpoint-specific NVP pickle files when present; otherwise compute 50 repeats for 1K, 10K, and 100K samples and save them.
- Allow forced NVP recomputation with one Boolean switch.
- Plot MC, CVAE, and CRealNVP mean estimates with 95% ranges and the existing benchmark on the same axes.
- Show vanilla options for a one-dimensional CVAE and barrier options for a CVAE with a running-minimum output, matching the existing robustness comparison convention.
- Fail clearly when a checkpoint, input result file, or incompatible checkpoint is encountered.

## Constraints

- Do not modify UI cell 19.
- Preserve unrelated notebook and worktree changes.
- Do not execute expensive GPU pricing during automated tests.
