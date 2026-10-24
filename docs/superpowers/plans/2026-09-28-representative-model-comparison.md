# Representative Model Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-method representative-parameter comparison to UI cell 20 of the CVAE result notebook.

**Architecture:** The notebook cell resolves a backend-specific NVP checkpoint, validates it against the current representative parameter vector, and loads or computes checkpoint-specific pricing results. It then combines those results with the MC and CVAE pickle files already used by UI cell 19 and renders the supported option family on shared comparison plots.

**Tech Stack:** Python, Jupyter notebook JSON, PyTorch, NumPy, Matplotlib, pickle, unittest

**Spec:** `docs/superpowers/specs/2026-09-28-representative-model-comparison.md`

## Global Constraints

- Modify UI cell 20, not UI cell 19.
- Preserve all unrelated dirty-worktree changes.
- Automated tests must not run real GPU inference.

## Review Focus

- Existing NVP pickle files must avoid model loading and inference.
- A missing cache must run the exact 1K/10K/100K by 50-repeat seed schedule and persist all supported payoff results.
- One-dimensional CVAE models must not show barrier options.
- BS/Heston aliases such as `_clip` and `_bbc` must resolve to the correct base backend.
- Incompatible checkpoint eta dimensions must fail before inference.

---

### Task 1: Representative MC/CVAE/CRealNVP Comparison Cell

**Files:**
- Modify: `paper/a_paper_cvae_result.ipynb` (UI cell 20)
- Create: `paper/tests/test_a_paper_cvae_result_representative_compare.py`

**Interfaces:**
- Consumes: notebook variables `cvae`, `model_type`, `eta_raw`, `result_base_path`, `multiple_suffix`, `B`, `K`, `r`, `T`, `validate_samples`, `valid_resample`, `mt_corr`, and `nvp_multiple`.
- Produces: `representative_comparison_figure`, `representative_comparison_axes`, `nvp_results_by_option`, and checkpoint-specific NVP pickle caches.

- [x] **Step 1: Write failing behavior tests**

Add tests that execute the real UI cell 20 source with temporary MC/CVAE/NVP pickle fixtures. Assert that cached results create two barrier panels containing MC, CVAE, and CRealNVP legend entries without loading the checkpoint. Add an uncached test with a fake `f_NVP` module and tiny sample/repeat overrides; assert deterministic seed use and saved pickle contents. Add a one-dimensional-CVAE test that produces only vanilla panels.

- [x] **Step 2: Run tests and verify RED**

Run: `python3 -m unittest paper.tests.test_a_paper_cvae_result_representative_compare -v`

Expected: FAIL because UI cell 20 is empty and does not produce the comparison outputs.

- [x] **Step 3: Implement UI cell 20**

Add editable checkpoint mappings, backend normalization, cache validation/generation, deterministic NVP pricing, persistence, calculation of means and 95% ranges, supported-option selection, and the combined plot.

- [x] **Step 4: Run focused and notebook tests**

Run: `python3 -m unittest paper.tests.test_a_paper_cvae_result_representative_compare paper.tests.test_a_paper_cvae_result_vanilla_compare -v`

Expected: PASS.

- [x] **Step 5: Run the repository test suite**

Run: `python3 -m unittest discover -s paper/tests -v`

Expected: New tests pass; report any pre-existing failures by name without modifying unrelated files.
