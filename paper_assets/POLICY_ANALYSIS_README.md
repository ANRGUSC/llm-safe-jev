# Extended Jev condition-2 vs condition-3 analysis

This directory is generated from cached experimental scores and decisions,
including the text-redacted public CSV. There are no API calls or new model outputs. It complements the original
`safety_paper_analysis.py` exports; its filenames are prefixed with `fig_`,
`tab_`, `curve_`, `grid_`, or `table_`.

## Scientific interpretation

- "Unsafe pass" is the fraction of **unsafe-labelled prompts with a non-refusal
  answer delivered**, not a validated harmful-completion rate. It is a proxy.
- "Safe suppressed" is the fraction of **XSTest safe** prompts for which no answer
  was delivered; includes baseline LLM refusals as well as safety-filter blocks.
  `xstest_safe_filter_block` measures only the additional blocks attributable to
  the safety filter, normalized by all benign prompts.
- AdvBench includes no safe prompts. Therefore `fig_frontier_advbench` pairs
  AdvBench unsafe prompts with XSTest safe prompts. XSTest-only frontier uses
  the XSTest unsafe and safe subsets, both from the same benchmark.
- Pooled micro: (AdvBench unsafe releases + XSTest unsafe releases)/(AdvBench
  unsafe N + XSTest unsafe N). This weights by number of prompts.
- Pooled macro: average of the two benchmark-specific unsafe-release fractions;
  it weights the benchmarks equally, regardless of sample sizes.
- The ALL-prompts micro error column gives an artificial benchmark-mix weighted
  result, not the real deployment error rate; safer to report the separate rates.
- Condition 2 sweeps **all observed prompt-score transitions**, plus reference
  thresholds. Condition 3 sweeps **three independently chosen thresholds**:
  low prompt score, high prompt score, and answer score, on a configurable
  grid (default step 0.05). Condition 3 can degenerate to prompt-only when low==high.
- Results named `in_sample` are **exploratory** because thresholds were selected
  and evaluated on the same data.
  For an initial check, `table_matched_caps_tune_then_heldout.csv` selects
  thresholds on a stratified 60% subset and evaluates on disjoint 40% prompts.
  A single prompt-level split does not control template or topical leakage,
  and is weaker than independent held-out benchmarks or repeated splits.
- The Pareto curves are discrete empirical points, not smooth continuous
  policies. The condition-3 frontier represents ONLY the candidate grid.
- The JEV cost axis counts hypothetical calls under routing; not measured
  latency, token cost, or paid API usage. All responses were originally
  generated for the base-model experiment, irrespective of counterfactual gating.
- Bootstrap confidence intervals for *original fixed thresholds* resample
  paired observations within datasets; no multiplicity corrections.

## Main analysis figures

1. `fig_four_view_frontier_comparison.pdf`: core apples-to-apples comparison
   (AdvBench, XSTest, pooled micro, pooled macro; XSTest-safe utility on x-axis).
2. `fig_matched_benign_refusal_budgets.pdf`: optimizing each condition to the
   same maximum allowable legitimate refusal rate (IN-SAMPLE).
3. `fig_condition2_full_threshold_sensitivity.pdf`: full 1-D threshold choice.
4. `fig_condition3_low_high_heatmaps.pdf`: joint low/high sensitivity at t_answer=0.5.
5. `fig_condition3_answer_threshold_sensitivity.pdf`: third threshold sensitivity.
6. `fig_heldout_matched_policy_budgets.pdf`: tentative holdout generalization.
7. `fig_incremental_review_benefit_vs_retuned_prompt_only.pdf`: marginal gain
   over prompt-only when both are reoptimized at the same legitimate-refusal cap.
8. `fig_matched_unsafe_release_budgets.pdf`: reverse matching on safety caps.
9. `fig_safety_vs_jev_calls_at_10pct_safe_budget.pdf`: safety versus review overhead.
10. `fig_benchmark_differences_between_unsafe_sets.pdf`: AdvBench vs XSTest divergence.

## Main LaTeX tables

- `tab_default_individual_pooled.tex`: default policies, separate and pooled.
- `tab_benign_filter_induced.tex`: legitimate refusal decomposition.
- `tab_paired_split_pooled.tex`: fixed-threshold paired bootstrap effects.
- `tab_matched_thresholds_pooled_micro.tex`: actual selected threshold triples.
- `tab_holdout_selected_policies.tex`: train-selected thresholds, held-out results.

Preamble: `\usepackage{booktabs,graphicx}`. Include with
`\input{paper_assets/tab_default_individual_pooled.tex}` and
`\includegraphics[width=\linewidth]{paper_assets/fig_four_view_frontier_comparison.pdf}`.

## Raw data and controls

- `curve_condition2_exact_thresholds.csv`: each exact prompt transition.
- `grid_condition3_joint_three_thresholds.csv`: every explored triple.
- `frontier_cond{2,3}_{advbench,xstest,pooled_micro,pooled_macro}_in_sample.csv`.
- `table_matched_benign_refusal_caps_in_sample.csv`: thresholds selected at common caps.
- `table_matched_caps_tune_then_heldout.csv`: train/test selection check.
- `table_optimal_thresholds_by_view_and_budget.csv`: threshold triples and costs.
- `table_incremental_benefit_of_retuned_answer_review.csv`: marginal value of
  answer review after a fair prompt-only retuning at each benign-refusal cap.
- `table_matched_unsafe_release_caps_in_sample.csv`: reverse matching on safety.
- `stratified_train_test_assignments.csv`: hashed prompt IDs and split assignments.
- `policy_analysis_manifest.json`: grids, benchmark counts, settings, input hash.
- `fig_prompt_roc_pr_xstest_vs_pooled.pdf`: input-score ROC/PR for XSTest-only and
  pooled (the latter is cross-dataset and has different positive prevalence).
- OPTIONAL `--annotations human_review_blinded_labeled.csv` enables an
  explicitly labeled-subset comparison. No unsafe-completion inference is made
  from JEV scores; annotator stratification biases unweighted sample rates.

**Important**: The exact scores and model generation are required to do
this analysis correctly. The repository includes the original cached numeric scores and outcomes in
a response-redacted 970-row CSV. This supports reproduction of numerical
policy comparisons without re-querying Jev. Redacted placeholders cannot be
used to independently judge response harmfulness. For that, authors must
use private original responses with a documented annotation procedure.
