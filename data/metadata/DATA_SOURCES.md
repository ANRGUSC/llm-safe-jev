# Historical data sources

- AdvBench: `walledai/AdvBench` on Hugging Face, split `train`, 520 prompt rows.
- XSTest: `walledai/XSTest`, split `test`, 250 safe and 200 unsafe prompts.
- Model: `Qwen/Qwen2.5-1.5B`, base checkpoint.
- Jev endpoint: `https://api.typesafe.ai/v1/systemone`, model `jev-latest`.
- Historical exact dataset commit IDs, Qwen checkpoint commit ID, Jev model build, Colab runtime package versions, and hardware details were **not recorded**.
- `original_results_sha256.txt` records the hash of the privately supplied original CSV. The original CSV itself is **not included**.
