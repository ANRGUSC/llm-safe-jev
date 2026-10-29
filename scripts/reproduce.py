"""One-command reproduction of quantitative results from public cached scores.

Run from repository root:
    python scripts/reproduce.py
No GPU, Jev key, Hugging Face download or model inference required.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def run(cmd):
    print('RUN:', ' '.join(map(str,cmd)),flush=True)
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,*map(str,cmd)],cwd=ROOT,env=env,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='data/derived/results_public.csv')
    parser.add_argument('--output',default='paper_assets')
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--policy-bootstrap',type=int,default=1200)
    parser.add_argument('--policy-grid',type=float,default=.05)
    parser.add_argument('--skip-figures',action='store_true')
    args=parser.parse_args()
    input_path=ROOT/args.input
    output=ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    run(['scripts/run_basic_analysis.py','--input',input_path,'--output',output,
         '--bootstraps',args.bootstrap])
    run(['scripts/run_policy_analysis.py','--input',input_path,'--output',output,
         '--bootstrap',args.policy_bootstrap,'--grid-step',args.policy_grid])
    run(['scripts/make_paper_tables.py','--input',input_path,'--output',ROOT/'paper/tables'])
    if not args.skip_figures:
        run(['scripts/make_paper_figures.py','--input',input_path,
             '--output',ROOT/'paper/figures','--assets',output])
    table=pd.read_csv(ROOT/'paper/tables/table1_primary_outcomes.csv')
    assert table.advbench_answered.tolist()==[349,2,3]
    assert table.xstest_unsafe_answered.tolist()==[178,15,37]
    assert table.xstest_safe_refused.tolist()==[3,27,10]
    audit=pd.read_csv(ROOT/'paper/tables/table5_audit_author_confirmed.csv')
    assert audit.loc[audit.dataset=='Total',['justified','unjustified','ambiguous']].values.tolist()==[[2,5,6]]
    print('\nPASS: all original-condition headline counts match the paper.')
    print('Quantitative exports:',output)
    print('Paper figures:',ROOT/'paper/figures')
    print('Paper tables:',ROOT/'paper/tables')
    print('Note: the AI-assisted audit aggregate was author-confirmed by manual inspection of all 13 responses; original row-level judgments are not available in this public package.')

if __name__=='__main__':main()
