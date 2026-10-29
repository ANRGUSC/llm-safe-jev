from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from jev_repro.safety_paper_analysis import load_results,evaluate
from jev_repro.safety_policy_optimization import PolicyData,read_input
ROOT=Path(__file__).resolve().parents[1]


def test_original_counts_and_redaction():
    path=ROOT/'data/derived/results_public.csv'
    df=load_results(path)
    assert df['set'].value_counts().to_dict()=={'advbench':520,'xs_safe':250,'xs_unsafe':200}
    assert df['prompt'].str.startswith('[REDACTED_PROMPT_').all()
    assert df['response'].str.startswith('[REDACTED_RESPONSE]').all()
    for c,col in [('llm','cond1'),('prompt','cond2'),('tier','cond3')]:
        d=evaluate(df,c)
        assert (d['answered']==df[col].eq('answered').to_numpy()).all()
    assert evaluate(df,'tier')['reviewed'].sum()==79
    assert evaluate(df,'tier')['answer_blocked'].sum()==13


def test_nested_policy_and_heldout():
    data=PolicyData(read_input(ROOT/'data/derived/results_public.csv'),seed=42,test_fraction=.4)
    for t in [.1,.44,.5,.765,.9]:
        a=data.decisions('cond2',t=t)
        b=data.decisions('cond3',t=.5,lo=t,hi=t)
        for key in ['released','prompt_blocked','reviewed','llm_called']:
            assert (a[key]==b[key]).all()
    assert data.test.sum()==388
    assert int((data.test&data.masks['xs_safe']).sum())==100


def test_threshold_tie_and_f1():
    data=PolicyData(read_input(ROOT/'data/derived/results_public.csv'),seed=42,test_fraction=.4)
    a=data.aggregate(data.decisions('cond2',t=.51))
    b=data.aggregate(data.decisions('cond2',t=__import__('numpy').nextafter(.51,1.)))
    assert round(a['xstest_safe_refusal']*250)==26
    assert round(b['xstest_safe_refusal']*250)==25
    assert round(b['pooled_unsafe_micro_pass']*720)==17
    import numpy as np
    d=data.decisions('cond3',t=.43,lo=.60,hi=.78)
    m=data.aggregate(d)
    assert round(m['pooled_unsafe_micro_pass']*720)==27
    assert round(m['xstest_safe_refusal']*250)==10
    assert round(m['answer_reviews_per_input']*970)==28
