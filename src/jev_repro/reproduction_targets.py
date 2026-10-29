"""Pinned *forward rerun targets*, not asserted historical model versions.

The 970-row dataset was produced with an unpinned Qwen repository and the
rolling `jev-latest` alias. These targets are selected for future runs and
are not evidence of the exact versions used for the original paper.
"""
QWEN_MODEL_ID = 'Qwen/Qwen2.5-1.5B'
QWEN_REVISION = '8faed761d45a263340a0528343f099c05c9a4323'
JEV_MODEL_ID = 'jev-1.14.0'
JEV_HISTORICAL_REQUESTED_ALIAS = 'jev-latest'
JEV_RELEASE_DATE = '2026-10-14'
QWEN_REVISION_SOURCE = 'https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B'
JEV_RELEASE_SOURCE = 'https://typesafe.ai/blog/jev-1-14'
