# Tested analysis environment

- Python: 3.13.5
- OS: Linux-6.18.55-x86_64-with-glibc2.41
- Package versions: see `requirements-tested.txt`.
- This is the environment in which the packaged offline scripts were tested, **not** the historical Qwen/Jev experiment environment.
- Running `pip install -r requirements-tested.txt` is an option for closer version matching; `requirements-analysis.txt` gives broader supported ranges.

## Pinned targets for a NEW live rerun

- Qwen model: `Qwen/Qwen2.5-1.5B`, commit `8faed761d45a263340a0528343f099c05c9a4323` (Hugging Face API snapshot consulted 2026-10-28).
- Jev model requested: `jev-1.14.0` (TypeSafe release 2026-10-14).
- Historical original requests: Qwen model without a revision and Jev `jev-latest` alias; **the exact historical Jev build was not recorded**.
- Fresh runs log requested versions in `private_runs/.../run_manifest.json` and returned Jev model identifiers in `jev_versions.json`, when provided by the service.
- Sources: https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B ; https://typesafe.ai/blog/jev-1-14

A pinned model ID does not freeze the provider's serving implementation or guarantee bit-for-bit identical inference.
