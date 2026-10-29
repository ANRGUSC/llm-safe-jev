"""Regenerate quantitative manuscript Tables 1, 2, 4, and 6.

Table 3 is a qualitative selection of benchmark prompts, not a statistical
export; Table 5's AI-assisted audit counts are reproduced only as *reported*
aggregates because the original individual audit annotations and judge setup
were not supplied. See docs/REPRODUCIBILITY_BOUNDARIES.md.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from jev_repro.safety_paper_analysis import load_results, evaluate
from jev_repro.safety_policy_optimization import PolicyData,read_input


def latex_table(path,caption,label,columns,rows):
    n=len(columns)
    tex=['\\begin{table}[t]','\\centering','\\footnotesize',
         '\\setlength{\\tabcolsep}{3pt}',f'\\caption{{{caption}}}',
         f'\\label{{{label}}}',
         '\\begin{tabular}{@{}l'+'r'*(n-1)+'@{}}',
         '\\toprule',' & '.join(columns)+' \\\\', '\\midrule']
    for row in rows:tex.append(' & '.join(map(str,row))+' \\\\')
    tex+=['\\bottomrule','\\end{tabular}','\\end{table}']
    path.write_text('\n'.join(tex)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/derived/results_public.csv')
    p.add_argument('--output',default='paper/tables')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    df=load_results(Path(args.input))
    groups=[('advbench','AdvBench'),('xs_unsafe','XSTest-unsafe'),('xs_safe','XSTest-safe')]
    decs={c:evaluate(df,c) for c in ['llm','prompt','tier']}
    rows=[]
    for c,title in [('llm','C1: LLM alone'),('prompt','C2: input gate'),('tier','C3: tiered gate')]:
        d=decs[c];n=[]
        for group,_ in groups:
            m=df['set'].eq(group).to_numpy()
            k=int(d['answered'][m].sum()) if group!='xs_safe' else int((~d['answered'][m]).sum())
            n.append(k)
        rows.append({'condition':title,'advbench_answered':n[0],
                     'xstest_unsafe_answered':n[1],
                     'pooled_unsafe_answered':n[0]+n[1],
                     'xstest_safe_refused':n[2]})
    pd.DataFrame(rows).to_csv(out/'table1_primary_outcomes.csv',index=False)
    latex_table(out/'table1_primary_outcomes.tex','Primary outcomes at the reported thresholds.',
                'tab:primary', ['Configuration','Adv. ans.','XSU ans.','Pooled ans.','XSS ref.'],
                [[r['condition'],f"{r['advbench_answered']}/520",
                  f"{r['xstest_unsafe_answered']}/200",f"{r['pooled_unsafe_answered']}/720",
                  f"{r['xstest_safe_refused']}/250"] for r in rows])
    dec=decs['tier'];breakdown=[]
    for key,name in groups+[('all','Total')]:
        m=np.ones(len(df),bool) if key=='all' else df['set'].eq(key).to_numpy()
        breakdown.append({'set':name,'input_blocked':int(dec['prompt_blocked'][m].sum()),
                          'llm_refused':int(dec['refused'][m].sum()),
                          'output_reviewed':int(dec['reviewed'][m].sum()),
                          'output_blocked':int(dec['answer_blocked'][m].sum()),
                          'answered':int(dec['answered'][m].sum())})
    pd.DataFrame(breakdown).to_csv(out/'table2_c3_decomposition.csv',index=False)
    latex_table(out/'table2_c3_decomposition.tex','C3 outcome decomposition. Output reviewed is an intermediate step.',
                'tab:decomp',['Set','In blk.','LLM ref.','Reviewed','Out blk.','Answered'],
                [[r['set'],r['input_blocked'],r['llm_refused'],r['output_reviewed'],
                  r['output_blocked'],r['answered']] for r in breakdown])
    # No-review C3: high input gate is unchanged, middle-tier answer is delivered.
    pscore=df.p_harm_prompt.to_numpy(float)
    baseline=df.cond1.eq('answered').to_numpy(bool)
    reviewed=dec['reviewed']
    no_review=(pscore<.9)&baseline
    ablation=[]
    for name,rel,rev in [('No output review',no_review,0),
                         ('With output review',dec['answered'],int(reviewed.sum()))]:
        row={'policy':name,'reviewed':rev}
        for key,_ in groups:
            mask=df['set'].eq(key).to_numpy()
            row[key+'_answered']=int(rel[mask].sum())
            row[key+'_refused']=int((~rel[mask]).sum())
        ablation.append(row)
    pd.DataFrame(ablation).to_csv(out/'table4_output_review_ablation.csv',index=False)
    latex_table(out/'table4_output_review_ablation.tex','C3 output-review ablation (counterfactual from cached scores).',
                'tab:ablation',['Policy','Adv. ans.','XSU ans.','XSS ref.','Rev.'],
                [[r['policy'],f"{r['advbench_answered']}/520",f"{r['xs_unsafe_answered']}/200",
                  f"{r['xs_safe_refused']}/250",r['reviewed']] for r in ablation])
    # AI-assisted audit: ONLY aggregate values copied from the manuscript.
    audit=[{'dataset':'AdvBench','justified':0,'unjustified':2,'ambiguous':0},
           {'dataset':'XSTest-safe','justified':0,'unjustified':3,'ambiguous':3},
           {'dataset':'XSTest-unsafe','justified':2,'unjustified':0,'ambiguous':3}]
    audit.append({'dataset':'Total','justified':2,'unjustified':5,'ambiguous':6})
    pd.DataFrame(audit).to_csv(out/'table5_audit_author_confirmed.csv',index=False)
    latex_table(out/'table5_audit_author_confirmed.tex',
                'AI-assisted audit of output blocks, manually confirmed by a paper author (row-level judgments unavailable).',
                'tab:audit',['Dataset','J','U','A','Total'],
                [[r['dataset'],r['justified'],r['unjustified'],r['ambiguous'],
                  r['justified']+r['unjustified']+r['ambiguous']] for r in audit])
    # Table 6 from explicit policy thresholds in the paper; held-out split seed=42.
    policy=PolicyData(read_input(Path(args.input)),seed=42,test_fraction=.4)
    settings=[('4% / all','C2 ($\\tau=.765$)','cond2',.765,None,None,'all'),
              ('4% / all','C3 ($.60/.80/.45$)','cond3',.45,.60,.80,'all'),
              ('10% / all','C2 ($\\tau=.51^{+}$)','cond2',np.nextafter(.51,1.),None,None,'all'),
              ('10% / all','C3 ($.45/.55/.15$)','cond3',.15,.45,.55,'all'),
              ('10% / test','C2 ($\\tau=.44$)','cond2',.44,None,None,'test'),
              ('10% / test','C3 ($.45/.45$, no rev.)','cond3',.5,.45,.45,'test')]
    matched=[]
    for budget,title,cond,t,lo,hi,subset in settings:
        d=policy.decisions(cond,t=t,lo=lo if lo is not None else .3,
                           hi=hi if hi is not None else .9)
        m=policy.aggregate(d,None if subset=='all' else policy.test)
        matched.append({'budget':budget,'policy':title,'unsafe_answer_rate':m['pooled_unsafe_micro_pass'],
                        'safe_refusal_rate':m['xstest_safe_refusal'],
                        'output_review_rate':m['answer_reviews_per_input']})
    pd.DataFrame(matched).to_csv(out/'table6_matched_policies.csv',index=False)
    latex_table(out/'table6_matched_policies.tex',
                'Matched XSTest-safe refusal budgets (in-sample and single held-out split).',
                'tab:matched',['Budget','Policy','H ans.','B ref.','Rev.'],
                [[r['budget'],r['policy'],f"{100*r['unsafe_answer_rate']:.1f}\\%",
                  f"{100*r['safe_refusal_rate']:.1f}\\%",
                  f"{100*r['output_review_rate']:.1f}\\%"] for r in matched])
    # Table 3: fixed manuscript-selected example row IDs. The public CSV has
    # redacted prompt text, so only scores and decisions can be regenerated.
    example_rows=[621,612,606,775,774,818]
    source=pd.read_csv(args.input,keep_default_na=False)
    examples=source[source.source_row.isin(example_rows)].copy()
    examples['manuscript_example_order']=examples.source_row.map(
        {row:i+1 for i,row in enumerate(example_rows)})
    examples=examples.sort_values('manuscript_example_order')
    examples[['manuscript_example_order','source_row','prompt_sha256','set',
              'p_harm_prompt','p_harm_answer','cond2','cond3']].to_csv(
                  out/'table3_example_indices_redacted.csv',index=False)
    print('Paper tables generated:',out)

if __name__=='__main__':main()
