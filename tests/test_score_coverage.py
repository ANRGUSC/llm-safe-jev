"""Prevent incomplete caches from silently supporting counterfactual sweeps."""
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jev_repro.safety_paper_analysis import load_results
from jev_repro.safety_policy_optimization import read_input


def test_output_scores_cover_all_original_tiers():
    df = read_input(ROOT / 'data/derived/results_public.csv')
    baseline = df.cond1.eq('answered')
    assert len(df) == 970
    assert baseline.sum() == 774
    for tier, count in [('pass', 214), ('review', 79), ('block', 481)]:
        selected = baseline & df.cond3_tier.eq(tier)
        assert selected.sum() == count
        assert df.loc[selected, 'p_harm_answer'].notna().all()
    assert df.loc[~baseline, 'p_harm_answer'].isna().all()


@pytest.mark.parametrize('loader', [load_results, read_input])
@pytest.mark.parametrize('tier', ['pass', 'block'])
def test_missing_score_outside_original_review_region_is_rejected(tmp_path, loader, tier):
    df = pd.read_csv(ROOT / 'data/derived/results_public.csv')
    index = df.index[df.cond1.eq('answered') & df.cond3_tier.eq(tier)][0]
    df.loc[index, 'p_harm_answer'] = float('nan')
    path = tmp_path / 'incomplete.csv'
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='(?i)(answer score|answer scores)'):
        loader(path)
