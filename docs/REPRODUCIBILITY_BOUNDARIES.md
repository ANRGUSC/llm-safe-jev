# What a reviewer can reproduce

| Paper item | Status | How |
|---|---|---|
| Three conditions, 970-prompt counts | **Exact from cached results** | `scripts/reproduce.py`, Table 1 |
| C3 input/refusal/review/block/delivered decomposition | **Exact from cached results** | Table 2 |
| Figure 1 architecture | **Recreated diagram** | `scripts/make_paper_figures.py` |
| Figure 2 Wilson intervals | **Numerically reproducible** | Figure 2 script, fixed original outcomes |
| Figure 3 C2 threshold sweep | **Exact score-based rates** | Figure 3 script (uses finer exact scores, not rounded published sweep) |
| Table 3 qualitative example prompts | **Manuscript-curated, not automatically regenerated from redacted text** | Refer to paper; private original results required for direct prompt text |
| Table 4 no-review ablation | **Exact cached counterfactual** | Table 4 script |
| Table 5 AI-assisted audit (2 justified / 5 unjustified / 6 ambiguous) | **Author-confirmed aggregate; not independently re-executable** | Table 5 CSV; Bhaskar reviewed all 13 and agreed with AI judgments and rationales; row-level annotations and judge details absent |
| Figure 4 matched C2/C3 frontiers | **Exact cached policies** | Figure 4 script; 0.05-grid C3 candidate set |
| Table 6 matched caps and 60/40 held-out comparison | **Exact cached policies** | Table 6 script; `seed=42`, stratified by dataset |
| Figure 5 C3 profiled macro-F1 heatmap | **Exact cached policies** | 0.01-grid, output threshold profiled; macro-F1 uses unsafe-block vs safe-release labels |
| Paired bootstrap, ROC/PR, calls | **Reproducible** | `paper_assets/` scripts |
| Fresh identical model generations/Jev API scores | **Not guaranteed** | Historical revision/versions unknown; forward reruns pin Qwen and Jev but may differ |
| Verified harmful assistance rates | **Not measured** | Would require independent response-level annotations |

The public dataset is intentionally **text-redacted**. Reproduction of the *numerical statistical analyses* does not require raw prompt/response text; rerunning inference or independently auditing output content does. Raw model responses are not distributed in this public package.

**Uncertainty caveat:** Wilson intervals are per-rate, and bootstrap CIs for fixed-policy paired differences are distinct from intervals for a threshold chosen on the same dataset. The C3 F1 maximum and in-sample frontier are exploratory. A single held-out split is a limited generalization check, not evidence of stable superiority.
