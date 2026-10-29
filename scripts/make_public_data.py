"""Create a public, response-redacted dataset from a PRIVATE original results.csv.

Never publish raw model responses without independent safety/rights review.
Hash columns allow local provenance checking but may be reversible by dictionary
lookup against the public benchmarks; they are NOT a privacy guarantee.
"""
import argparse
import hashlib
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Private original results.csv')
    p.add_argument('--output',default='data/derived/results_public.csv')
    args=p.parse_args()
    src=Path(args.input);out=Path(args.output)
    if src.resolve()==out.resolve():raise ValueError('Refusing to overwrite original results')
    df=pd.read_csv(src,keep_default_na=False,dtype={'prompt':str,'response':str})
    for key in ['set','prompt','response','p_harm_prompt','p_harm_answer','cond1','cond2','cond3']:
        if key not in df:raise ValueError(f'Missing column: {key}')
    df.insert(0,'source_row',range(len(df)))
    df.insert(1,'prompt_sha256',[hashlib.sha256(s.encode('utf-8')).hexdigest() for s in df.prompt])
    df.insert(2,'response_sha256',[hashlib.sha256(s.encode('utf-8')).hexdigest() for s in df.response])
    df['prompt']=[f'[REDACTED_PROMPT_{i:04d}]' for i in range(len(df))]
    df['response']=['[REDACTED_RESPONSE]' if s.strip() else '' for s in df.response]
    out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    print(f'Wrote {len(df)} redacted rows to {out}; verify before public release.')

if __name__=='__main__':main()
