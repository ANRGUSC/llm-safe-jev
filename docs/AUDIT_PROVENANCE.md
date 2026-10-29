# AI-assisted audit: author-confirmed judgments

The paper reports an exploratory AI-assisted review of **all 13 responses** blocked by the Condition 3 output-review stage. **Bhaskar Krishnamachari reports personally inspecting each of the 13 original responses and confirming agreement with the AI-generated judgment and rationale for every response** (author confirmation provided 28 October 2026). This is a human confirmation of the 13 assessments, not a second independent blinded annotation study.

| Benchmark | Justified | Unjustified | Ambiguous | Total |
|---|---:|---:|---:|---:|
| AdvBench | 0 | 2 | 0 | 2 |
| XSTest-safe | 0 | 3 | 3 | 6 |
| XSTest-unsafe | 2 | 0 | 3 | 5 |
| **Total** | **2** | **5** | **6** | **13** |

The aggregate results are preserved in `paper/tables/table5_audit_author_confirmed.csv` and `.tex`. The individual AI judge model, exact judge prompt, row-level judgments and written rationales have **not been supplied to this repository**, so this table is a documented, author-confirmed result rather than an independently re-executable audit. We do not invent row-level labels or claim that a second independent human rater reproduced them.

`data/derived/output_block_audit_template.csv` identifies the 13 blocked records by dataset, row index, hashes, and Jev scores, without distributing raw responses. Its row-level `audit_label` and `auditor_notes` fields remain blank until the original annotations can be supplied and approved for public release.

**Distinct quantities:** 13 is the number of responses blocked by Jev's output reviewer; the original LLM refusal detector contains 14 substring patterns.
