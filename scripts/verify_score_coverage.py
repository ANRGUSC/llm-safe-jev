"""Verify that cached output scores support changing the C3 review region."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jev_repro.safety_policy_optimization import read_input


def verify(path):
    # The existing loader rejects missing non-refusal output scores and invalid scores.
    df = read_input(path)
    baseline = df.cond1.eq('answered')
    p = df.p_harm_prompt
    tiers = {'low': p < .30, 'intermediate': (p >= .30) & (p < .90),
             'high': p >= .90}
    rows = []
    for cohort in ('advbench', 'xs_unsafe', 'xs_safe'):
        for tier, mask in tiers.items():
            selected = df['set'].eq(cohort) & mask
            needed = selected & baseline
            rows.append(dict(cohort=cohort, original_tier=tier,
                             prompts=int(selected.sum()),
                             nonrefusals=int(needed.sum()),
                             output_scores=int(df.loc[needed, 'p_harm_answer'].notna().sum())))
    # Checking all possible reviewable rows proves coverage for every output
    # threshold. Also explicitly check every input-boundary pair in the fine grid.
    grid = np.round(np.linspace(0, 1, 101), 2)
    pairs = 0
    for i, low in enumerate(grid):
        for high in grid[i:]:
            reviewed = baseline & (p >= low) & (p < high)
            if df.loc[reviewed, 'p_harm_answer'].isna().any():
                raise ValueError(f'Missing scores for review region [{low}, {high})')
            pairs += 1
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'PASS: {int(baseline.sum())}/{int(baseline.sum())} non-refusal responses scored; '
          f'{int((~baseline).sum())} baseline refusals never require review.')
    print(f'PASS: {pairs} input-boundary pairs (0.01 grid), each supporting every output threshold.')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/derived/results_public.csv')
    verify(parser.parse_args().input)
