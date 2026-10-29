# Historical experiment versus pinned forward rerun

| Component | Original recorded experiment | Selected fresh-rerun target |
|---|---|---|
| Qwen | `Qwen/Qwen2.5-1.5B` (no commit recorded) | `8faed761d45a263340a0528343f099c05c9a4323` |
| Jev | `jev-latest` (rolling alias; resolved build not saved) | `jev-1.14.0` (released 2026-10-14) |
| Datasets | `walledai/AdvBench` train and `walledai/XSTest` test (no commits saved) | same sources, unless `--dataset-revision` supplied |

The original runs occurred over approximately the last two weeks of October 2026, overlapping TypeSafe's rollout from Jev 1.13 to 1.14. **It is not possible to determine which version served the original requests from the existing CSV or notebook.** Assigning `jev-1.14.0` to the original data would be misleading.

For a **new** run, the portable runner defaults to the pinned targets shown above. It records requested identifiers and any provider-returned model identifiers. It does not overwrite the original notebook's constants. Cached original scores, not newly generated scores, reproduce the published figures and tables.

Sources: [Qwen Hugging Face API](https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B), [TypeSafe Jev 1.14 announcement](https://typesafe.ai/blog/jev-1-14).
