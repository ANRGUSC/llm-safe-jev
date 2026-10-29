# Paper-ready Jev safety experiment outputs

These files were generated **offline from cached experimental scores and decisions** by `safety_paper_analysis.py`.
The analysis performs no external API calls. The repository includes a text-redacted version of the original results.

## LaTeX integration

LaTeX preamble: `\usepackage{booktabs,graphicx}`.

```
\input{paper_assets/tab_headline.tex}
\input{paper_assets/tab_prompt_discrimination.tex}
\input{paper_assets/tab_paired_comparison.tex}
\input{paper_assets/tab_inference_cost.tex}
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{paper_assets/fig_safety_utility_tradeoff.pdf}
  \caption{In-sample policy tradeoff. The axes report *proxy* error rates; parameter sweeps do not establish held-out improvements.}
  \label{fig:tradeoff}
\end{figure}
```

Alternatively insert one of the other `fig_*.pdf` graphics. High-resolution PNG versions are included for slides/previews.

## Main exports

- `tab_headline.tex`: exactly counted three-cohort headline table.
- `tab_paired_comparison.tex`: **paired** effect of changing prompt-only to tiered, with 95% paired bootstrap CIs.
- `tab_prompt_discrimination.tex`: XSTest-only input-score ROC--AUC and average precision.
- `tab_inference_cost.tex`: counterfactual LLM and JEV *call counts* (not latency or dollars).
- `tab_answer_review_ablation.tex`: incremental benefit/cost of second-stage answer review.
- `tab_matched_operating_point.tex`: in-sample prompt-only retuning to the same benign suppression as tiered policy (exploratory).
- `tab_prompt_threshold_sweep.tex`: selected threshold choices, for an appendix.
- `fig_headline_proxy_rates.pdf`: bad-event rates with Wilson binomial CIs.
- `fig_prompt_roc_pr.pdf`: ROC and precision--recall, both with XSTest labels.
- `fig_prompt_threshold_sensitivity.pdf`: prompt-only score threshold sweep.
- `fig_prompt_score_ecdf.pdf`: score distributions per benchmark.
- `fig_safety_utility_tradeoff.pdf`: comparison of prompt-only and tiered families.
- `fig_tiered_routing_outcomes.pdf`: routing decomposition, release/refusal/block.
- `table_condition_metrics.csv`: full counts, Wilson CIs and per-set calls.
- `table_paired_comparisons.csv`: paired changes, discordant cases, bootstrap CIs and exact McNemar p-values.
- `table_inference_call_proxies.csv`: condition-level simulated invocation counts.
- `table_answer_review_ablation.csv`: how many extra JEV calls blocked outputs in each benchmark.
- `matched_operating_point_exploratory.json`: matched benign-suppression comparison.
- `sweep_prompt_threshold.csv` / `sweep_tier_parameters.csv`: all sweep settings.
- `pareto_frontier_exploratory.csv`: nondominated configurations (IN-SAMPLE ONLY).
- `analysis_manifest.json`: seed, environment and SHA-256 of exact input CSV.
- `human_review_blinded.csv` / `human_review_routing_key.csv`: separate human-labeling sheet + hidden routing map.

## Interpretation and methodological scope

1. **Non-refusal is not validated harmful content**. The 'AdvBench/XSTest-unsafe answered' numbers are *proxy outcomes*, not attack-success rates or validated harmful completions. A baseline string matcher can mislabel refusals and benign discussions. Human annotation is recommended.
2. **XSTest-safe suppressed** combines refusal by the base LLM and blocking by JEV. This is a false-suppression / overrefusal *proxy*, not evidence that all blocked answers would be useful.
3. **JEV scores are uncalibrated for this analysis**. No probabilistic calibration claim is made; the analysis reports discrimination and threshold behavior.
4. **ROC--AUC and average precision** use XSTest safe/unsafe gold *prompt labels*. AdvBench is excluded to avoid conflating two benchmarks. Average precision changes with the positive fraction.
5. **95% Wilson intervals** summarize prompt-level binomial sampling uncertainty; **paired bootstrap intervals** resample rows *within each benchmark*. Shared templates and semantic duplication may create correlation not captured by these intervals; treat results as descriptive for these fixed benchmarks.
6. **All policy sweeps are in-sample**. A chosen threshold or Pareto configuration must be validated on held-out prompts to claim generalization. No hypothesis tests adjusted for multiple policy searches.
7. **Inference-call proxies are hypothetical**: during the original experiment all LLM outputs were generated for all prompts, even when a deployed filter would block them. They do not measure wall time, token generation, endpoint costs, or production response latency.
8. **Human-review sheet** uses stratified sampling by dataset and observed tiered-policy outcome. Its unweighted sample fraction does *not* estimate population prevalence. Annotators should label the response against the prompt independently of Jev score. If possible, annotate the full dataset or use inverse-stratum-weighted estimates.
9. The base model is `Qwen/Qwen2.5-1.5B` with simple `User: ... Assistant:` prompting and string-match refusal detection. Results may differ with instruction-tuned/chat-template models and stricter response-based safety labeling.
10. **No new JEV queries are needed** for these diagnostics. The script requires complete per-row cached scores/outcomes (the released redacted CSV is sufficient), not only rounded summary rates.

## Additional response annotation

Independent response-harmfulness evaluation requires access to the original (non-redacted) text, a documented annotation rubric, and an adjudication procedure. A blinded review design can use the routing key separately from the annotation worksheet. These supplementary outputs contain redacted placeholders when generated from the public CSV.
