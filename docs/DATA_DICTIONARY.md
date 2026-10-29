# Public derived data dictionary

`data/derived/results_public.csv` contains 970 rows in **the original experiment's row order**, produced by `scripts/make_public_data.py` from the privately held `results.csv`.

| Column | Meaning |
|---|---|
| `source_row` | 0-based original row index; preserves stratified split and bootstrap order |
| `prompt_sha256` | SHA-256 of original UTF-8 prompt, for provenance (not privacy-preserving against dictionary matching) |
| `response_sha256` | SHA-256 of original UTF-8 response, for provenance |
| `set` | `advbench`, `xs_safe`, `xs_unsafe` |
| `prompt` | **Redacted placeholder**, NOT original prompt; retained because analysis functions require field |
| `response` | **Redacted placeholder**, NOT original response; refusal decision saved in `cond1` |
| `p_harm_prompt` | Original Jev input score |
| `p_harm_answer` | Original Jev output score, blank for baseline refusals |
| `cond3_tier` | Original C3 routing: `block`, `review`, `pass` |
| `cond1` | Original baseline terminal outcome |
| `cond2` | Original C2 terminal outcome |
| `cond3` | Original C3 terminal outcome |

Placeholders must **never** be sent to Qwen/Jev. The numerical analyses use cohort, scores, and decisions, not literal prompt/response text. The general analysis module's `human_review_blinded.csv` is **not suitable for annotation** when generated from this redacted input.

For a fresh experiment, download original benchmarks and run the experiment with your own API credentials, then redact private results before sharing. Verify benchmark redistribution terms and whether hash publication is appropriate.
