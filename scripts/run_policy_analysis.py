import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jev_repro.safety_policy_optimization import run_analysis
p=argparse.ArgumentParser()
p.add_argument('--input',required=True);p.add_argument('--output',required=True)
p.add_argument('--bootstrap',type=int,default=1200)
p.add_argument('--grid-step',type=float,default=.05)
a=p.parse_args()
run_analysis(a.input,a.output,seed=42,grid_step=a.grid_step,test_fraction=.4,
             bootstrap=a.bootstrap,annotations_csv=None)
