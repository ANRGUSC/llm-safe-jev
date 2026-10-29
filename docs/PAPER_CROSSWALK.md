# Paper figures and tables: source crosswalk

This index maps paper figures and tables to their reproducible artifacts and generating code. The generated files may differ in typography from the typeset paper.

| Manuscript item | Generated artifact | Code |
|---|---|---|
| Fig. 1: C1/C2/C3 pipeline | `paper/figures/fig1_pipeline.pdf` | `scripts/make_paper_figures.py:fig1` |
| Fig. 2: Wilson intervals | `paper/figures/fig2_wilson.pdf` | `fig2` |
| Fig. 3: C2 threshold sweep | `paper/figures/fig3_c2_threshold.pdf` | `fig3` |
| Fig. 4: C2/C3 matched frontiers | `paper/figures/fig4_matched_frontiers.pdf` | `fig4` |
| Fig. 5: profiled C3 macro-F1 heatmap | `paper/figures/fig5_c3_profiled_f1.pdf` | `fig5` |
| Table 1: primary counts | `paper/tables/table1_primary_outcomes.tex` | `scripts/make_paper_tables.py` |
| Table 2: C3 decomposition | `paper/tables/table2_c3_decomposition.tex` | same |
| Table 3: illustrative prompt examples | manuscript only (not generated from redacted data) | source prompts in private CSV |
| Table 4: output-review ablation | `paper/tables/table4_output_review_ablation.tex` | same |
| Table 5: AI-assisted audit | `paper/tables/table5_audit_author_confirmed.tex` | author-confirmed aggregate, no row-level judge metadata |
| Table 6: matched budgets/holdout | `paper/tables/table6_matched_policies.tex` | same |
| Supplementary figures/tables | `paper_assets/` | `src/jev_repro/safety_*analysis.py`, `safety_policy_optimization.py` |

**Figure 5 optimization:** End-to-end macro-F1 treats a harmful-labeled prompt as correctly handled when **not answered**, and an XSTest-safe prompt as correctly handled when **answered**. It is a proxy objective, not a harmful-output evaluation. Each heatmap cell `(low,high)` takes the best score over output thresholds `a` at increments of 0.01. All three thresholds were selected on the same data; 0.9512 is in-sample.

**Held-out split:** the original row order and seed 42 are preserved in the public CSV. The split is 60% train / 40% test, stratified by the three cohorts. Policies are tuned on train and evaluated once on test. The reported test results at the 10% training-set safe-refusal cap are 2.8% pooled unsafe-prompt answers and 14.0% safe refusals for both C2 and C3, with no answer review used by the selected C3 policy.

**Threshold boundary semantics:** The cached scores contain ties at exactly 0.51. With `block if score >= τ`, the exact cutoff `τ=0.51` suppresses 26/250 safe prompts (10.4%). A cutoff immediately above 0.51 (`np.nextafter(.51, 1.)`, denoted `0.51+` in the generated table) suppresses 25/250 (10.0%). Reported rounded thresholds should be interpreted with this boundary convention.

**Table 3 audit trail:** `paper/tables/table3_example_indices_redacted.csv` maps the six examples in their manuscript order to original row indices, SHA-256 prompt hashes, Jev scores and C2/C3 decisions, without repeating the prompt text or any model completions. These six examples were selected manually for the manuscript; their text is already in the paper.
