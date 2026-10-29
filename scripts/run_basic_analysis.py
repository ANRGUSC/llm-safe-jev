import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jev_repro.safety_paper_analysis import analyze
p=argparse.ArgumentParser()
p.add_argument('--input',required=True);p.add_argument('--output',required=True)
p.add_argument('--bootstraps',type=int,default=2000)
a=p.parse_args()
analyze(a.input,a.output,threshold=.5,low=.3,high=.9,seed=42,bootstraps=a.bootstraps)
