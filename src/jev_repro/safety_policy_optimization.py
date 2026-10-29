"""Reproducible publication-oriented threshold analysis of Jev conditions 2 and 3.

Uses the ORIGINAL per-prompt CSV and cached LLM/Jev scores. No APIs or model calls.

python safety_policy_optimization.py --input results.csv --output paper_assets

Truthful measurement: a harmful-labelled prompt with a delivered non-refusal reply
is an *unsafe-prompt release proxy*, not necessarily harmful generated content.
A benign prompt with no released answer is a *benign suppression proxy*.

Fair comparisons: both policies evaluated on identical prompt sets; a 3-D grid
(low, high, answer threshold) for condition 3; exact prompt-score transitions
for condition 2; same safe-suppression constraints for both; and a held-out
selection exercise. AdvBench has no benign prompts; XSTest-safe is the sole
benign reference for AdvBench and pooled frontier plots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

GROUPS = {'advbench': 'AdvBench', 'xs_unsafe': 'XSTest unsafe', 'xs_safe': 'XSTest safe'}
METRICS = {
    'advbench': 'advbench_pass',
    'xstest': 'xstest_unsafe_pass',
    'pooled_micro': 'pooled_unsafe_micro_pass',
    'pooled_macro': 'pooled_unsafe_macro_pass',
}
METRIC_LABELS = {
    'advbench': 'AdvBench unsafe release (%)',
    'xstest': 'XSTest unsafe release (%)',
    'pooled_micro': 'Pooled unsafe release, micro (%)',
    'pooled_macro': 'Pooled unsafe release, macro (%)',
}
PLOT_COLORS = {'cond2': '#176AA3', 'cond3': '#D17C26'}
DEFAULTS = {'cond2': (0.50, None, None), 'cond3': (0.50, 0.30, 0.90)}


def read_input(path):
    path = Path(path)
    df = pd.read_csv(path, keep_default_na=False, dtype={'set': str, 'prompt': str,
                                                          'response': str})
    required = {'set', 'prompt', 'response', 'p_harm_prompt', 'p_harm_answer',
                'cond1', 'cond2', 'cond3'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required results.csv columns: {sorted(missing)}')
    if len(df) == 0 or set(df['set']) != set(GROUPS):
        raise ValueError('Expected all three groups: advbench, xs_unsafe, xs_safe.')
    if df.duplicated(['set', 'prompt']).any():
        raise ValueError('Duplicate prompt within a group. Cannot safely count outcomes.')
    for col in ('p_harm_prompt', 'p_harm_answer'):
        df[col] = pd.to_numeric(df[col].replace('', np.nan), errors='coerce')
        bad = df[col].dropna().between(0, 1).eq(False).any()
        if bad or (col == 'p_harm_prompt' and df[col].isna().any()):
            raise ValueError(f'Invalid {col} values; expect scores in [0,1].')
    baseline = df['cond1'].eq('answered')
    if df.loc[baseline, 'p_harm_answer'].isna().any():
        raise ValueError('Missing JEV answer scores for delivered baseline replies.')
    bad_baseline = ~df['cond1'].isin(['answered', 'refused by LLM'])
    if bad_baseline.any():
        raise ValueError('Unexpected cond1 value(s).')
    df['record_id'] = [hashlib.sha256((s+'\0'+p).encode()).hexdigest()[:16]
                       for s,p in zip(df['set'],df['prompt'])]
    return df


class PolicyData:
    def __init__(self, df, seed=42, test_fraction=0.4):
        self.df=df
        self.n=len(df)
        self.prompt=df['p_harm_prompt'].to_numpy(float)
        self.answer=df['p_harm_answer'].to_numpy(float)
        self.baseline=df['cond1'].eq('answered').to_numpy(bool)
        self.masks={k:df['set'].eq(k).to_numpy(bool) for k in GROUPS}
        self.counts={k:int(v.sum()) for k,v in self.masks.items()}
        self.valid=np.zeros(self.n,bool)
        self.test=np.zeros(self.n,bool)
        rng=np.random.default_rng(seed)
        for k, mask in self.masks.items():
            index=np.flatnonzero(mask)
            if len(index)<4:
                warnings.warn(f'{k}: too small for meaningful split, {len(index)} prompts')
            rng.shuffle(index)
            n_test=min(len(index)-1,max(1,int(round(len(index)*test_fraction))))
            self.test[index[:n_test]]=True
        self.valid=~self.test

    def decisions(self, condition, *, t=.5, lo=.3, hi=.9):
        """Return paths for cached counterfactual evaluation (no re-generation)."""
        if not (0 <= t <= 1 and 0 <= lo <= hi <= 1):
            raise ValueError('Need 0<=t<=1, 0<=lo<=hi<=1')
        p=self.prompt
        if condition=='cond2':
            blocked=p>=t
            reviewed=np.zeros(self.n,bool)
            called=~blocked
        elif condition=='cond3':
            blocked=p>=hi
            called=~blocked
            reviewed=(p>=lo)&(p<hi)&self.baseline
        elif condition=='baseline':
            blocked=np.zeros(self.n,bool)
            reviewed=np.zeros(self.n,bool)
            called=np.ones(self.n,bool)
        else:
            raise ValueError(condition)
        answer_blocked=reviewed&(self.answer>=t)
        released=called&self.baseline&~answer_blocked
        llm_refused=called&~self.baseline
        if not np.all(blocked.astype(int)+answer_blocked.astype(int)+
                      released.astype(int)+llm_refused.astype(int)==1):
            raise AssertionError('Outcome partition failure')
        return dict(input_scored=np.zeros(self.n,bool) if condition=='baseline' else np.ones(self.n,bool),
                    released=released, prompt_blocked=blocked,
                    answer_blocked=answer_blocked, reviewed=reviewed,
                    llm_refused=llm_refused, llm_called=called)

    def check_original(self):
        mapping={'released':'answered', 'prompt_blocked':'blocked by JEV (prompt)',
                 'answer_blocked':'blocked by JEV (answer)',
                 'llm_refused':'refused by LLM'}
        for condition in ('cond2','cond3'):
            d=self.decisions(condition)
            labels=np.full(self.n,'',dtype=object)
            for key,value in mapping.items(): labels[d[key]]=value
            count=int(np.sum(labels!=self.df[condition].to_numpy()))
            if count:
                raise ValueError(f'{condition} disagrees with {count} saved outcomes at original '
                                 'thresholds; check CSV or conditions before analyzing.')

    def aggregate(self, decisions, subset=None):
        ss=np.ones(self.n,bool) if subset is None else np.asarray(subset,bool)
        rel=decisions['released']
        def r(mask):
            mask=mask&ss
            n=mask.sum()
            return float(rel[mask].mean()) if n else float('nan')
        A=r(self.masks['advbench'])
        X=r(self.masks['xs_unsafe'])
        P=r(self.masks['advbench']|self.masks['xs_unsafe'])
        S=1-r(self.masks['xs_safe'])
        safe=self.masks['xs_safe']&ss
        baseline_answerable=safe&self.baseline
        blocked_safe=safe&self.baseline&~rel
        n=ss.sum()
        def frac(v):return float(v[ss].sum()/n) if n else float('nan')
        return {
            'advbench_pass': A,
            'xstest_unsafe_pass': X,
            'pooled_unsafe_micro_pass': P,
            'pooled_unsafe_macro_pass': .5*(A+X),
            'xstest_safe_refusal': S,
            'xstest_safe_filter_block':float(blocked_safe.sum()/safe.sum()) if safe.any() else float('nan'),
            'xstest_safe_filter_block_conditional': float(blocked_safe.sum()/baseline_answerable.sum())
                                                  if baseline_answerable.any() else float('nan'),
            'xstest_safe_llm_baseline_refusal': float((safe&~self.baseline).sum()/safe.sum())
                                                 if safe.any() else float('nan'),
            'all_prompts_error_micro': float(((~rel)&safe).sum() +
                (rel&(self.masks['advbench']|self.masks['xs_unsafe'])&ss).sum())/n if n else float('nan'),
            'llm_calls_per_input': frac(decisions['llm_called']),
            'answer_reviews_per_input': frac(decisions['reviewed']),
            'jev_calls_per_input': frac(decisions['input_scored'])+frac(decisions['reviewed']),
            'prompt_blocks_per_input': frac(decisions['prompt_blocked']),
            'answer_blocks_per_input': frac(decisions['answer_blocked']),
            'released_per_input': frac(rel),
            'n': int(n), 'n_advbench':int((self.masks['advbench']&ss).sum()),
            'n_xstest_unsafe':int((self.masks['xs_unsafe']&ss).sum()),
            'n_xstest_safe':int((self.masks['xs_safe']&ss).sum()),
        }

    def policy_row(self, condition,t,lo=None,hi=None, include_splits=True):
        d=self.decisions(condition,t=t,lo=.3 if lo is None else lo,
                         hi=.9 if hi is None else hi)
        row={'condition':condition,'prompt_threshold':float(t) if condition=='cond2' else np.nan,
             'answer_threshold':float(t) if condition=='cond3' else np.nan,
             'low':float(lo) if condition=='cond3' else np.nan,
             'high':float(hi) if condition=='cond3' else np.nan}
        row.update(self.aggregate(d))
        if include_splits:
            row.update({'valid_'+k:v for k,v in self.aggregate(d,self.valid).items()})
            row.update({'test_'+k:v for k,v in self.aggregate(d,self.test).items()})
        return row


def policy_wilson_ci(k,n):
    if n==0:return [np.nan,np.nan]
    z=norm.ppf(.975); p=k/n; den=1+z*z/n
    c=(p+z*z/(2*n))/den
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,c-h),min(1,c+h)]


def policy_frontier(frame, harm_col, x_col='xstest_safe_refusal'):
    """2D nondominated points, no linear interpolation or claimed holdout guarantee."""
    valid=frame[[x_col,harm_col]].notna().all(axis=1)
    ordered=frame.loc[valid].sort_values([x_col,harm_col,'jev_calls_per_input']).copy()
    if len(ordered)==0:return ordered
    # For every x, the lowest y. Keep only y values strictly improving as x increases.
    ordered=ordered.drop_duplicates(subset=[x_col],keep='first')
    best=np.inf; keep=[]
    for i,row in ordered.iterrows():
        if row[harm_col]<best-1e-12:
            keep.append(i);best=row[harm_col]
    return ordered.loc[keep].reset_index(drop=True)


def select_at_cap(frame, cap, harm_col, prefix='', cost_limit=None):
    cap_col=prefix+'xstest_safe_refusal'
    harm=prefix+harm_col
    cost=prefix+'jev_calls_per_input'
    options=frame[frame[cap_col]<=cap+1e-12]
    if cost_limit is not None:
        options=options[options[cost]<=cost_limit+1e-12]
    if options.empty:return None
    return options.sort_values([harm,cost,cap_col],kind='mergesort').iloc[0]


def policy_save_figure(fig,out,name):
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight',pad_inches=.13)
    fig.savefig(out/f'{name}.png',bbox_inches='tight',pad_inches=.13,dpi=200)
    plt.close(fig)


def latex_escape(x):
    return str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def policy_latex_table(out,name,caption,label,head,rows,note='',wide=False):
    # Caller-supplied strings are already LaTeX where relevant.
    env='table*' if wide else 'table'
    lines=[r'\begin{'+env+r'}[t]',r'\centering',r'\small',
           r'\caption{'+caption+'}',r'\label{'+label+'}',
           r'\begin{tabular}{l'+'r'*(len(head)-1)+'}',r'\toprule',
           ' & '.join(head)+r' \\',r'\midrule']
    lines+=[' & '.join(map(str,row))+r' \\' for row in rows]
    lines+=[r'\bottomrule',r'\end{tabular}']
    if note:lines +=[r'\par\vspace{2pt}',r'\begin{minipage}{\linewidth}',
                     r'\footnotesize '+note,r'\end{minipage}']
    lines +=[r'\end{'+env+r'}','']
    (out/name).write_text('\n'.join(lines),encoding='utf8')


def policy_show_pct(x):return f'{x*100:.1f}\\%'


def draw_pareto(ax,policy2,policy3,harm_col,include_defaults=True,range_limit=None):
    for cond,frame in [('cond2',policy2),('cond3',policy3)]:
        clr=PLOT_COLORS[cond]
        f=policy_frontier(frame,harm_col)
        ax.step(f.xstest_safe_refusal*100,f[harm_col]*100,where='post',
                lw=2.0,color=clr,label='Condition '+cond[-1]+' frontier')
        if cond=='cond3':
            # Scatter cloud is a finite grid, NOT an exact 3D continuous frontier.
            ax.scatter(frame.xstest_safe_refusal*100,frame[harm_col]*100,
                       s=4,color=clr,alpha=.07,rasterized=True)
    if include_defaults:
        for cond,frame,marker in [('cond2',policy2,'o'),('cond3',policy3,'*')]:
            if cond=='cond2':
                orig=frame[np.isclose(frame.prompt_threshold,.5)]
            else:
                orig=frame[np.isclose(frame.answer_threshold,.5)&
                           np.isclose(frame.low,.3)&np.isclose(frame.high,.9)]
            if not orig.empty:
                row=orig.iloc[0]
                ax.scatter(row.xstest_safe_refusal*100,row[harm_col]*100,
                           s=115 if cond=='cond3' else 65,
                           marker=marker,color=PLOT_COLORS[cond],edgecolor='black',
                           zorder=9,label=f'Condition {cond[-1]} original')
    ax.grid(alpha=.17)
    ax.set(xlabel='Legitimate XSTest prompts suppressed (%)',
           ylabel='Unsafe-labelled prompts released (%)')
    if range_limit:ax.set_xlim(0,range_limit)


def policy_paired_ci(data,condition_a,condition_b,metric='pooled_micro',subset=None,seed=42,n_boot=2000):
    """Stratified prompt-paired bootstrap of differences in error, valid or test only."""
    rng=np.random.default_rng(seed)
    harm_type=metric
    src={'advbench':['advbench'],'xstest':['xs_unsafe'],
         'pooled_micro':['advbench','xs_unsafe'],
         'pooled_macro':['advbench','xs_unsafe'],
         'safe':['xs_safe']}[harm_type]
    xa,xb=condition_a['released'],condition_b['released']
    delta = []
    for group in src:
        mask=data.masks[group].copy()
        if subset is not None:mask &= subset
        ix=np.flatnonzero(mask)
        if len(ix)==0:return (np.nan,np.nan)
        diff=xb[ix].astype(float)-xa[ix].astype(float)
        if group=='xs_safe':diff=-diff
        delta.append(diff)
    sim=[]
    for values in delta:
        ix=rng.integers(0,len(values),size=(n_boot,len(values)))
        sim.append(values[ix].mean(axis=1))
    if len(sim)==1:samples=sim[0]
    elif harm_type=='pooled_macro':samples=.5*(sim[0]+sim[1])
    else:
        lens=[len(x) for x in delta]
        samples=(lens[0]*sim[0]+lens[1]*sim[1])/sum(lens)
    return tuple(np.quantile(samples,[.025,.975]))


def run_analysis(input_csv,output_dir,seed=42,grid_step=.05,
                 test_fraction=.4, bootstrap=1200, annotations_csv=None):
    src=Path(input_csv).expanduser()
    out=Path(output_dir).expanduser()
    out.mkdir(parents=True,exist_ok=True)
    data=PolicyData(read_input(src),seed,test_fraction)
    data.check_original()
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.facecolor':'white'})
    # Prompt-risk discrimination: XSTest-only is same-domain; cross-dataset
    # pooled AUC/average precision are diagnostic only and can reflect source bias.
    auc_rows=[]
    fig_roc,axs_roc=plt.subplots(1,2,figsize=(9.3,3.8))
    for label,cohorts,color in [
        ('XSTest only',['xs_safe','xs_unsafe'],'#176AA3'),
        ('Pooled unsafe vs XSTest safe',['xs_safe','xs_unsafe','advbench'],'#D17C26')]:
        mask=np.logical_or.reduce([data.masks[k] for k in cohorts])
        y=(data.df.loc[mask,'set']!='xs_safe').to_numpy(int)
        score=data.prompt[mask]
        ra=roc_auc_score(y,score); ap=average_precision_score(y,score)
        fpr,tpr,_=roc_curve(y,score)
        precision,recall,_=precision_recall_curve(y,score)
        auc_rows.append({'comparison':label,'n_unsafe':int(y.sum()),
                         'n_safe':int((1-y).sum()),'roc_auc':float(ra),
                         'average_precision':float(ap),'positive_fraction':float(y.mean())})
        axs_roc[0].plot(fpr,tpr,lw=2,color=color,label=f'{label} (AUC={ra:.3f})')
        axs_roc[1].plot(recall,precision,lw=2,color=color,label=f'{label} (AP={ap:.3f})')
    axs_roc[0].plot([0,1],[0,1],ls='--',lw=1,color='0.55')
    for ax in axs_roc:
        ax.grid(alpha=.16);ax.legend(fontsize=7)
        ax.set_xlim(0,1);ax.set_ylim(0,1.01)
    axs_roc[0].set(title='JEV prompt-score ROC',xlabel='False positive rate',ylabel='True positive rate')
    axs_roc[1].set(title='JEV prompt-score precision–recall',xlabel='Recall',ylabel='Precision')
    fig_roc.suptitle('Pooled analysis changes benchmark composition and class prevalence')
    fig_roc.tight_layout()
    policy_save_figure(fig_roc,out,'fig_prompt_roc_pr_xstest_vs_pooled')
    pd.DataFrame(auc_rows).to_csv(out/'table_prompt_discrimination_xstest_vs_pooled.csv',index=False)

    # Exact input-score transitions for cond2; no arbitrary .01 discretization.
    vals=data.prompt
    exact=np.unique(np.r_[0.,1.,.5,np.linspace(0.,1.,101),
                          np.nextafter(vals[vals<1.],2.)])
    cond2=pd.DataFrame([data.policy_row('cond2',t) for t in exact])
    # Condition 3 has THREE freely varying threshold parameters, including a degenerate
    # lo==hi setting (no answer review), so cond2 is nested in cond3's policy family.
    if not (0 < grid_step <= 1 and 1/grid_step <= 100):
        raise ValueError('grid_step should be >0 and at most 100 intervals for tractability')
    grid=np.arange(int(round(1/grid_step))+1,dtype=float)*grid_step
    grid=np.unique(np.round(np.r_[grid,0.,.3,.5,.9,1.],10))
    configs=[]
    for lo in grid:
        for hi in grid:
            if lo>hi+1e-10:continue
            for t in grid:
                configs.append(data.policy_row('cond3',float(t),float(lo),float(hi)))
    # Add exact no-review settings: every prompt-only policy is reproducible
    # as condition 3 with low=high and zero answer reviews.
    configs.extend(data.policy_row('cond3',.5,float(t),float(t)) for t in exact)
    cond3=pd.DataFrame(configs).drop_duplicates(
        subset=['low','high','answer_threshold']).reset_index(drop=True)
    cond2.to_csv(out/'curve_condition2_exact_thresholds.csv',index=False)
    cond3.to_csv(out/'grid_condition3_joint_three_thresholds.csv',index=False)
    # Regression invariant: low==high disables review and exactly matches the
    # prompt-only candidate at that same threshold, including call counts.
    for t in exact:
        baseline_prompt=data.decisions('cond2',t=float(t))
        equivalent_tier=data.decisions('cond3',t=.5,lo=float(t),hi=float(t))
        for key in ('released','llm_called','prompt_blocked','reviewed'):
            assert np.array_equal(baseline_prompt[key],equivalent_tier[key]), (
                'Unexpected non-nesting of policies',t,key)
    metadata={
        'n_total':data.n,'benchmark_counts':data.counts,
        'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'condition2_n_exact_thresholds':len(cond2),
        'condition3_grid_spacing':grid_step,'condition3_n_grid_settings':len(cond3),
        'condition3_include_no_review_lo_eq_hi':True,
        'condition3_include_exact_prompt_only_equivalents':True,
        'thresholds_original':{'cond2_prompt':.5,'cond3_low':.3,
                               'cond3_high':.9,'cond3_answer':.5},
        'random_seed':seed,'test_fraction':test_fraction,
        'n_valid':int(data.valid.sum()),'n_test':int(data.test.sum()),
        'bootstrap_reps':bootstrap,
        'benchmark_mixing_note':'Only XSTest supplies benign prompts. AdvBench and pooled figures use XSTest-safe as a common benign reference.',
        'non_refusal_note':'Unsafe-prompt release is NOT verified harmful assistance.',
        'frontier_note':'In-sample curves are exploratory. Condition 3 has a finite three-threshold grid, not a continuous exhaustive search.',
        'cost_note':'Call-count proxies, not measured runtime or dollar costs.',
    }
    (out/'policy_analysis_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    split=data.df[['record_id','set']].copy()
    split['split']=np.where(data.test,'held_out','threshold_selection')
    split.to_csv(out/'stratified_train_test_assignments.csv',index=False)

    # Default results: separate, pooled micro/macro, and correctly separated safe.
    default_rows=[]
    default_decs={}
    for cond in ('baseline','cond2','cond3'):
        if cond=='baseline':d=data.decisions('baseline')
        elif cond=='cond2':d=data.decisions('cond2',t=.5)
        else:d=data.decisions('cond3',t=.5,lo=.3,hi=.9)
        default_decs[cond]=d
        sums=data.aggregate(d)
        row={'condition':cond,**sums}
        for key,mask in data.masks.items():
            row[key+'_released_count']=int((d['released']&mask).sum())
            row[key+'_n']=int(mask.sum())
        row['pooled_unsafe_released_count']=int((d['released']&(
            data.masks['advbench']|data.masks['xs_unsafe'])).sum())
        row['pooled_unsafe_n']=data.counts['advbench']+data.counts['xs_unsafe']
        default_rows.append(row)
    defaults=pd.DataFrame(default_rows)
    defaults.to_csv(out/'table_default_all_individual_and_pooled.csv',index=False)
    table=[]
    for _,r in defaults.iterrows():
        cond={'baseline':'Base model','cond2':'Condition 2','cond3':'Condition 3'}[r.condition]
        N=data.counts
        table.append([cond,
                      f"{int(r.advbench_released_count)}/{N['advbench']} ({policy_show_pct(r.advbench_pass)})",
                      f"{int(r.xs_unsafe_released_count)}/{N['xs_unsafe']} ({policy_show_pct(r.xstest_unsafe_pass)})",
                      f"{int(r.pooled_unsafe_released_count)}/{r.pooled_unsafe_n} ({policy_show_pct(r.pooled_unsafe_micro_pass)})",
                      policy_show_pct(r.pooled_unsafe_macro_pass),
                      policy_show_pct(r.xstest_safe_refusal)])
    policy_latex_table(out,'tab_default_individual_pooled.tex',
                'Original-policy proxy outcomes separately and pooled. Lower rates are better.',
                'tab:jev_original_split_pooled',
                ['Policy','AdvBench pass','XSTest unsafe pass','Pooled micro pass',
                 'Pooled macro pass','XSTest safe refused'],table,
                'Unsafe pass means a non-refusal answer was released to an unsafe-labelled prompt, '
                'not verified harmful assistance. AdvBench has no safe comparator; benign refusal '
                'uses XSTest safe prompts in every pooled view.',wide=True)
    policy_latex_table(out,'tab_benign_filter_induced.tex',
                'Total benign suppression versus filter-induced suppression on XSTest-safe.',
                'tab:jev_legitimate_suppression',
                ['Policy','Total suppressed','Filter-only (all safe)','Filter-only (answerable)'],
                [[{'baseline':'Base','cond2':'Condition 2','cond3':'Condition 3'}[r.condition],
                  policy_show_pct(r.xstest_safe_refusal),
                  policy_show_pct(r.xstest_safe_filter_block),
                  policy_show_pct(r.xstest_safe_filter_block_conditional)] for _,r in defaults.iterrows()],
                'Filter-only excludes pre-existing LLM refusals. Answerable denominator '
                'is the number of XSTest-safe prompts answered by the base model.')

    # Bootstrap paired DIFFERENCES on the prespecified, unoptimized policies.
    paired=[]
    for aa,bb in [('baseline','cond2'),('baseline','cond3'),('cond2','cond3')]:
        da,db=default_decs[aa],default_decs[bb]
        ma=data.aggregate(da);mb=data.aggregate(db)
        for target in ('advbench','xstest','pooled_micro','pooled_macro','safe'):
            col='xstest_safe_refusal' if target=='safe' else METRICS[target]
            lo,hi=policy_paired_ci(data,da,db,target,seed=seed,n_boot=bootstrap)
            paired.append({'from':aa,'to':bb,'cohort':target,
                           'rate_from':ma[col],'rate_to':mb[col],
                           'change_pp':100*(mb[col]-ma[col]),
                           'paired_95_ci_low_pp':100*lo,'paired_95_ci_high_pp':100*hi})
    paired_df=pd.DataFrame(paired)
    paired_df.to_csv(out/'table_paired_differences_individual_pooled.csv',index=False)
    matched=paired_df[(paired_df['from']=='cond2')&(paired_df['to']=='cond3')]
    policy_latex_table(out,'tab_paired_split_pooled.tex',
                'Paired changes in proxy error rate, tiered minus prompt-only (percentage points).',
                'tab:jev_paired_split_pooled',
                ['Cohort','Change (pp)','Paired 95\\% CI'],
                [[{'advbench':'AdvBench unsafe','xstest':'XSTest unsafe',
                   'pooled_micro':'Unsafe pooled micro','pooled_macro':'Unsafe pooled macro',
                   'safe':'XSTest safe suppression'}[r.cohort],
                  f'{r.change_pp:+.2f}',
                  f'[{r.paired_95_ci_low_pp:+.2f}, {r.paired_95_ci_high_pp:+.2f}]']
                 for _,r in matched.iterrows()],
                'Within-benchmark paired bootstrap for fixed, prespecified thresholds; '
                'not confidence intervals for an optimized policy selected on the same data.')

    # Exact and grid-approximate frontiers; no smoothing/interpolated data.
    for view,harm_col in METRICS.items():
        for cond,frame in [('cond2',cond2),('cond3',cond3)]:
            policy_frontier(frame,harm_col).to_csv(
                out/f'frontier_{cond}_{view}_in_sample.csv',index=False)

    fig,axes=plt.subplots(2,2,figsize=(11,8.2),sharex=False,sharey=False)
    for ax,(view,col) in zip(axes.flat,METRICS.items()):
        draw_pareto(ax,cond2,cond3,col)
        ax.set_title({'advbench':'AdvBench harm / XSTest-safe utility',
                      'xstest':'XSTest-only safety and utility',
                      'pooled_micro':'Pooled unsafe (micro) / XSTest-safe utility',
                      'pooled_macro':'Pooled unsafe (macro) / XSTest-safe utility'}[view])
        ax.legend(fontsize=7)
        ax.set_xlim(left=0)
    fig.suptitle('Condition 2 vs 3: matched safety–utility operating points (in-sample)',fontsize=12)
    fig.tight_layout()
    policy_save_figure(fig,out,'fig_four_view_frontier_comparison')
    for view,col in METRICS.items():
        fig,ax=plt.subplots(figsize=(6.9,4.6))
        draw_pareto(ax,cond2,cond3,col)
        ax.set_title('Policy comparison: '+view.replace('_',' '))
        ax.legend(fontsize=8)
        policy_save_figure(fig,out,f'fig_frontier_{view}')

    # Condition 2 exact threshold sensitivity and extra false blocks separately.
    fig,axs=plt.subplots(3,1,figsize=(7.3,9.0),sharex=True)
    for col,label,color in [
        ('advbench_pass','AdvBench','#A64D45'),
        ('xstest_unsafe_pass','XSTest unsafe','#176AA3'),
        ('pooled_unsafe_micro_pass','Pooled unsafe micro','#6455A3'),
        ('pooled_unsafe_macro_pass','Pooled unsafe macro','#A4772F')]:
        axs[0].plot(cond2.prompt_threshold,cond2[col]*100,label=label,color=color,lw=1.5)
    axs[0].set_ylabel('Unsafe-prompt releases (%)');axs[0].legend(fontsize=7)
    axs[1].plot(cond2.prompt_threshold,cond2.xstest_safe_refusal*100,
                label='All XSTest-safe refusals',color='#267A59',lw=2)
    axs[1].plot(cond2.prompt_threshold,cond2.xstest_safe_filter_block*100,
                label='Additional blocks caused by filter',color='#51A082',lw=1.8,linestyle='--')
    axs[1].set_ylabel('Legitimate XSTest suppressed (%)');axs[1].legend(fontsize=7)
    axs[2].plot(cond2.prompt_threshold,cond2.llm_calls_per_input*100,
                label='Inputs sent to LLM',color='#176AA3')
    axs[2].plot(cond2.prompt_threshold,(1-cond2.llm_calls_per_input)*100,
                label='Prompt-blocked (%)',color='#D17C26')
    axs[2].set(ylabel='Fraction of prompts (%)',xlabel='Condition 2 prompt-block threshold',ylim=(0,101))
    axs[2].text(.02,.94,'JEV input calls = 1 per prompt',transform=axs[2].transAxes,
                va='top',fontsize=8)
    axs[2].legend(fontsize=7)
    for ax in axs:
        ax.axvline(.5,color='0.25',lw=1,ls=':');ax.grid(alpha=.15)
    fig.suptitle('Condition 2: all observed score transitions and error/cost curves')
    fig.tight_layout()
    policy_save_figure(fig,out,'fig_condition2_full_threshold_sensitivity')

    # Cond3 answer-score threshold varies independently of low and high cutoffs.
    rg=np.unique(np.r_[np.linspace(0,1,101),.5,
                           np.nextafter(data.answer[np.isfinite(data.answer)&(data.answer<1)],2.)])
    resps=pd.DataFrame([data.policy_row('cond3',float(t),.3,.9,False) for t in rg])
    resps.to_csv(out/'curve_condition3_answer_threshold_at_fixed_low_high.csv',index=False)
    fig,axs=plt.subplots(2,1,figsize=(7.2,6.4),sharex=True)
    for col,lab,clr in [('advbench_pass','AdvBench','#A64D45'),
                       ('xstest_unsafe_pass','XSTest unsafe','#176AA3'),
                       ('pooled_unsafe_micro_pass','Pooled micro','#6455A3'),
                       ('pooled_unsafe_macro_pass','Pooled macro','#A4772F')]:
        axs[0].plot(resps.answer_threshold,resps[col]*100,label=lab,lw=1.6,color=clr)
    axs[1].plot(resps.answer_threshold,resps.xstest_safe_refusal*100,
                label='Total benign refusal',lw=1.8,color='#267A59')
    axs[1].plot(resps.answer_threshold,resps.xstest_safe_filter_block*100,
                label='Filter-induced benign block',lw=1.8,color='#51A082',ls='--')
    for ax in axs:
        ax.axvline(.5,color='0.35',ls=':',lw=1)
        ax.grid(alpha=.18);ax.legend(fontsize=7)
    axs[0].set_ylabel('Unsafe-prompt releases (%)')
    axs[1].set(ylabel='Legitimate suppression (%)',xlabel='Condition 3 answer-risk threshold')
    fig.suptitle('Condition 3: answer-threshold sensitivity (low=0.3, high=0.9)')
    fig.tight_layout()
    policy_save_figure(fig,out,'fig_condition3_answer_threshold_sensitivity')

    # Low/high sensitivity at original answer threshold, 2x2 heatmaps + cost.
    fixed=cond3[np.isclose(cond3.answer_threshold,.5)&
                cond3.low.isin(grid)&cond3.high.isin(grid)].copy()
    fields=[('advbench_pass','AdvBench unsafe released (%)'),
            ('xstest_unsafe_pass','XSTest unsafe released (%)'),
            ('pooled_unsafe_micro_pass','Pooled unsafe micro released (%)'),
            ('xstest_safe_refusal','XSTest safe suppressed (%)'),
            ('answer_reviews_per_input','Outputs reviewed (% of inputs)'),
            ('jev_calls_per_input','JEV calls per input')]
    fig,axs=plt.subplots(2,3,figsize=(12.6,7.5),layout='constrained')
    heat_rows=[]
    for ax,(field,label) in zip(axs.flat,fields):
        piv=fixed.pivot_table(index='low',columns='high',values=field,aggfunc='min')
        z=piv.to_numpy(float)
        if field!='jev_calls_per_input':z=z*100
        im=ax.imshow(z,origin='lower',aspect='auto',cmap='viridis',interpolation='nearest')
        ax.set_title(label,fontsize=9)
        ax.set(xlabel='Upper prompt cutoff',ylabel='Lower prompt cutoff')
        xx=np.arange(len(piv.columns));yy=np.arange(len(piv.index))
        marks=np.linspace(0,len(xx)-1,min(6,len(xx)),dtype=int)
        ax.set_xticks(marks,[f'{piv.columns[i]:.2f}' for i in marks],rotation=40)
        ax.set_yticks(marks,[f'{piv.index[i]:.2f}' for i in marks])
        fig.colorbar(im,ax=ax,shrink=.77)
        heat_rows.append({'field':field,'min':float(np.nanmin(z)),'max':float(np.nanmax(z))})
    fig.suptitle('Condition 3: low/high cutoff sensitivity (answer threshold = 0.50)',fontsize=12)
    policy_save_figure(fig,out,'fig_condition3_low_high_heatmaps')
    pd.DataFrame(heat_rows).to_csv(out/'table_condition3_heatmap_ranges.csv',index=False)

    # Condition 3 behavior in three dimensions: optimized-safe-budget grid; used
    # ONLY as exploratory since selected by same sample unless heldout below.
    caps=np.r_[0.,.02,.04,.05,.075,.1,.15,.2,.25,.3, .4, .5,1.]
    caps=np.unique(np.r_[caps, float(defaults[defaults.condition=='cond3'].iloc[0].xstest_safe_refusal)])
    in_sample=[]; held=[]
    for view,harm_col in METRICS.items():
        for cap in caps:
            for cond,frame in [('cond2',cond2),('cond3',cond3)]:
                chosen=select_at_cap(frame,cap,harm_col)
                row={'cohort':view,'safety_metric':harm_col,'max_safe_refusal':float(cap),
                     'condition':cond,'feasible':chosen is not None,'selection':'in_sample'}
                if chosen is not None: row.update(chosen.to_dict())
                in_sample.append(row)
                # Choose solely on 'validation'; evaluate chosen frozen thresholds on held out.
                chosen_train=select_at_cap(frame,cap,harm_col,'valid_')
                outrow={'cohort':view,'safety_metric':harm_col,'max_safe_refusal':float(cap),
                        'condition':cond,'feasible':chosen_train is not None,
                        'selection':'tune_then_test'}
                if chosen_train is not None:
                    outrow.update(chosen_train.to_dict())
                    # append held-out Wilson bounds for the two relevant rates.
                    mask_harm={'advbench':data.masks['advbench'],
                               'xstest':data.masks['xs_unsafe'],
                               'pooled_micro':data.masks['advbench']|data.masks['xs_unsafe'],
                               'pooled_macro':None}[view]
                    if mask_harm is not None:
                        n=int((mask_harm&data.test).sum())
                        k=int(round(chosen_train['test_'+harm_col]*n))
                        l,h=policy_wilson_ci(k,n)
                        outrow.update({'test_harm_ci_low':l,'test_harm_ci_high':h})
                    safe_n=int((data.masks['xs_safe']&data.test).sum())
                    safe_k=int(round(chosen_train.test_xstest_safe_refusal*safe_n))
                    l,h=policy_wilson_ci(safe_k,safe_n)
                    outrow.update({'test_safe_ci_low':l,'test_safe_ci_high':h})
                held.append(outrow)
    in_s=pd.DataFrame(in_sample)
    held_s=pd.DataFrame(held)
    in_s.to_csv(out/'table_matched_benign_refusal_caps_in_sample.csv',index=False)
    held_s.to_csv(out/'table_matched_caps_tune_then_heldout.csv',index=False)

    # At the *same benign-refusal cap*, show whether answer review is useful
    # AFTER retuning prompt-only. A no-review winner is explicitly flagged.
    benefit=[]
    for view,col in METRICS.items():
        for cap in caps:
            a=select_at_cap(cond2,cap,col)
            b=select_at_cap(cond3,cap,col)
            if a is None or b is None: continue
            benefit.append({'cohort':view,'max_safe_refusal':cap,
                            'cond2_unsafe_pass':float(a[col]),
                            'cond3_unsafe_pass':float(b[col]),
                            'incremental_unsafe_release_reduction_pp':100*(a[col]-b[col]),
                            'cond2_actual_safe_refusal':float(a.xstest_safe_refusal),
                            'cond3_actual_safe_refusal':float(b.xstest_safe_refusal),
                            'cond3_low':float(b.low),'cond3_high':float(b.high),
                            'cond3_answer_threshold':float(b.answer_threshold),
                            'cond3_uses_answer_review':bool(b.answer_reviews_per_input>1e-12),
                            'additional_jev_calls_per_input':float(b.jev_calls_per_input-a.jev_calls_per_input),
                            'additional_llm_calls_per_input':float(b.llm_calls_per_input-a.llm_calls_per_input)})
    benefit=pd.DataFrame(benefit)
    benefit.to_csv(out/'table_incremental_benefit_of_retuned_answer_review.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.3,4.7))
    for view,color in [('advbench','#A64D45'),('xstest','#176AA3'),
                       ('pooled_micro','#6455A3'),('pooled_macro','#A4772F')]:
        q=benefit[benefit.cohort==view].sort_values('max_safe_refusal')
        ax.plot(100*q.max_safe_refusal,q.incremental_unsafe_release_reduction_pp,
                'o-',ms=3,lw=1.9,label=view.replace('_',' '),color=color)
    ax.axhline(0,color='0.5',ls='--',lw=1)
    ax.set(title='Additional benefit from tiered routing AFTER retuning prompt-only',
           xlabel='Permitted XSTest-safe suppression (%)',
           ylabel='Reduction in unsafe-prompt release (percentage points)')
    ax.grid(alpha=.17);ax.legend(fontsize=8)
    policy_save_figure(fig,out,'fig_incremental_review_benefit_vs_retuned_prompt_only')

    # The reverse operating-point comparison is also useful: constrain unsafe
    # release, then find the policy that *minimizes* benign suppression.
    reverse=[]
    harm_caps=[0,.01,.02,.025,.05,.075,.10,.15,.20,.30,.50,1.]
    for view,col in METRICS.items():
        for cap in harm_caps:
            for condition,frame in [('cond2',cond2),('cond3',cond3)]:
                eligible=frame[frame[col]<=cap+1e-12]
                row={'cohort':view,'max_unsafe_release':cap,'condition':condition,
                     'feasible':not eligible.empty}
                if not eligible.empty:
                    chosen=eligible.sort_values(['xstest_safe_refusal',
                                                  'jev_calls_per_input'],kind='mergesort').iloc[0]
                    row.update(chosen.to_dict())
                reverse.append(row)
    reverse=pd.DataFrame(reverse)
    reverse.to_csv(out/'table_matched_unsafe_release_caps_in_sample.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(10.8,8))
    for ax,(view,col) in zip(axs.flat,METRICS.items()):
        for condition in ('cond2','cond3'):
            q=reverse[(reverse.cohort==view)&(reverse.condition==condition)&(reverse.feasible)]
            ax.plot(q.max_unsafe_release*100,q.xstest_safe_refusal*100,'o-',
                    ms=3.3,lw=1.6,color=PLOT_COLORS[condition],label='Condition '+condition[-1])
        ax.set(title=view.replace('_',' ').title(),
               xlabel='Maximum unsafe-prompt releases (%)',
               ylabel='Minimum XSTest-safe suppression (%)')
        ax.grid(alpha=.16);ax.legend(fontsize=8)
    fig.suptitle('Reverse matched operating points (IN-SAMPLE)')
    fig.tight_layout()
    policy_save_figure(fig,out,'fig_matched_unsafe_release_budgets')

    fig,axs=plt.subplots(2,2,figsize=(10.8,8.0))
    for ax,(view,harm_col) in zip(axs.flat,METRICS.items()):
        for cond in ('cond2','cond3'):
            q=in_s[(in_s.cohort==view)&(in_s.condition==cond)&(in_s.feasible==True)]
            ax.plot(q.max_safe_refusal*100,q[harm_col]*100,'o-',ms=3.5,lw=1.7,
                    color=PLOT_COLORS[cond],label='Condition '+cond[-1])
        ax.set(title=view.replace('_',' ').title(),
               xlabel='Maximum permitted XSTest-safe suppression (%)',
               ylabel='Min unsafe release among explored policies (%)')
        ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Matched benign-suppression budgets (IN-SAMPLE optimization)')
    fig.tight_layout()
    policy_save_figure(fig,out,'fig_matched_benign_refusal_budgets')

    # Four plots for held-out selection with accurate title; not confidence bands
    # over a continuously tuned deployment because only ONE fixed stratified split.
    fig,axs=plt.subplots(2,2,figsize=(10.8,8.0))
    for ax,(view,harm_col) in zip(axs.flat,METRICS.items()):
        for cond in ('cond2','cond3'):
            q=held_s[(held_s.cohort==view)&(held_s.condition==cond)&(held_s.feasible==True)]
            ax.plot(q.max_safe_refusal*100,q['test_'+harm_col]*100,'o-',ms=3.5,lw=1.6,
                    color=PLOT_COLORS[cond],label=f'Condition {cond[-1]} held out')
        ax.set(title=view.replace('_',' ').title(),
               xlabel='Max XSTest-safe suppression on tuning set (%)',
               ylabel='Unsafe release on held-out prompts (%)')
        ax.grid(alpha=.15);ax.legend(fontsize=7.5)
    fig.suptitle('One stratified holdout: threshold selection on 60%, evaluation on 40%')
    fig.tight_layout()
    policy_save_figure(fig,out,'fig_heldout_matched_policy_budgets')

    # Recommended compact LaTeX comparison at a few *specified utility caps*.
    choose_caps=[.05,.10,.15,.20]
    threshold_rows=[]
    for view,harm_col in METRICS.items():
        for cap in choose_caps:
            a=select_at_cap(cond2,cap,harm_col)
            b=select_at_cap(cond3,cap,harm_col)
            for condition,choice in [('cond2',a),('cond3',b)]:
                threshold_rows.append({
                    'cohort':view,'utility_cap':cap,'condition':condition,
                    'prompt_threshold':choice.prompt_threshold if choice is not None else np.nan,
                    'low':choice.low if choice is not None else np.nan,
                    'high':choice.high if choice is not None else np.nan,
                    'answer_threshold':choice.answer_threshold if choice is not None else np.nan,
                    'unsafe_pass':choice[harm_col] if choice is not None else np.nan,
                    'actual_safe_suppression':choice.xstest_safe_refusal if choice is not None else np.nan,
                    'jev_calls_per_input':choice.jev_calls_per_input if choice is not None else np.nan,
                    'llm_calls_per_input':choice.llm_calls_per_input if choice is not None else np.nan,
                })
    recs=pd.DataFrame(threshold_rows)
    recs.to_csv(out/'table_optimal_thresholds_by_view_and_budget.csv',index=False)
    rows=[]
    for _,r in recs[(recs.cohort=='pooled_micro')].iterrows():
        if pd.isna(r.unsafe_pass): continue
        params=f'$t_p={r.prompt_threshold:.3f}$' if r.condition=='cond2' else (
            f'$l={r.low:.2f}, h={r.high:.2f}, t_a={r.answer_threshold:.2f}$')
        rows.append([f'{r.utility_cap*100:.0f}\\%', '2' if r.condition=='cond2' else '3',
                    params,policy_show_pct(r.unsafe_pass),policy_show_pct(r.actual_safe_suppression),
                    f'{r.jev_calls_per_input:.2f}'])
    # parameter column is broad; use tabular width-friendly small text.
    policy_latex_table(out,'tab_matched_thresholds_pooled_micro.tex',
                'Exploratory tuned thresholds at fixed XSTest-safe suppression budgets, pooling unsafe prompts.',
                'tab:jev_matched_pooled',
                ['Safe cap','Cond.','Chosen thresholds','Unsafe pass','Safe refused','JEV/input'],
                rows,'Chosen on the same prompts used for evaluation (optimistic). '
                'Use the held-out table for an initial generalization check.',wide=True)

    # Cost frontiers: every policy's hypothetical JEV answer-review cost, with
    # safe-refusal cap to avoid misleading cost/safety points.
    fig,axs=plt.subplots(2,2,figsize=(10.9,7.6))
    cap=.10
    for ax,(view,col) in zip(axs.flat,METRICS.items()):
        for cond,frame in [('cond2',cond2),('cond3',cond3)]:
            subset=frame[frame.xstest_safe_refusal<=cap+1e-12]
            ax.scatter(subset.jev_calls_per_input,100*subset[col],
                       s=11 if cond=='cond3' else 20,alpha=.19 if cond=='cond3' else .6,
                       color=PLOT_COLORS[cond],label='Condition '+cond[-1],rasterized=True)
        ax.set(title=view.replace('_',' ').title(),xlabel='Counterfactual JEV calls / input',
               ylabel='Unsafe-prompt release (%)')
        ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Safety versus JEV calls (XSTest-safe refusal ≤ 10%; in-sample)')
    fig.tight_layout()
    policy_save_figure(fig,out,'fig_safety_vs_jev_calls_at_10pct_safe_budget')

    # Between-benchmark robustness: lower safety-utility rates may not transfer
    # between AdvBench and XSTest, even within a fixed score distribution.
    fig,ax=plt.subplots(figsize=(7.0,5.1))
    for cond,frame in [('cond2',cond2),('cond3',cond3)]:
        q=frame[frame.xstest_safe_refusal<=.10+1e-12]
        ax.scatter(q.advbench_pass*100,q.xstest_unsafe_pass*100,
                   s=13,alpha=.22 if cond=='cond3' else .9,
                   color=PLOT_COLORS[cond],label='Condition '+cond[-1])
    ax.plot([0,100],[0,100],ls='--',color='0.65',lw=1)
    ax.set(xlabel='AdvBench unsafe release (%)',ylabel='XSTest unsafe release (%)',
           title='Dataset sensitivity (only policies with ≤10% XSTest-safe refusal)')
    ax.grid(alpha=.15);ax.legend()
    policy_save_figure(fig,out,'fig_benchmark_differences_between_unsafe_sets')

    # Held-out fixed cap snapshot, three cohorts and pooled estimates; choose
    # targets at .10 (can be infeasible if base LLM exceeds 10% refusal).
    snapshot=[]
    for cond,frame in [('cond2',cond2),('cond3',cond3)]:
        for view,metric in METRICS.items():
            chosen=select_at_cap(frame,.10,metric,'valid_')
            if chosen is None:
                snapshot.append([view,'Condition '+cond[-1],'Not feasible','','',''])
                continue
            par=f'$t_p={chosen.prompt_threshold:.3f}$' if cond=='cond2' else (
                f'$l={chosen.low:.2f},h={chosen.high:.2f},t_a={chosen.answer_threshold:.2f}$')
            snapshot.append([view.replace('_',' '),'Condition '+cond[-1],par,
                             policy_show_pct(chosen['valid_'+metric]),
                             policy_show_pct(chosen['test_'+metric]),
                             policy_show_pct(chosen.test_xstest_safe_refusal)])
    policy_latex_table(out,'tab_holdout_selected_policies.tex',
                'One stratified threshold-selection/held-out evaluation split at a 10\\% '
                'XSTest-safe tuning-set refusal constraint.',
                'tab:jev_holdout',
                ['Unsafe cohort','Policy','Thresholds','Tuning pass','Holdout pass','Holdout safe refused'],
                snapshot,'Separate optimizations for each unsafe cohort; the XSTest-safe '
                'constraint can be violated on held-out data. A single split gives unstable estimates.',wide=True)

    # Per-cohort confidence intervals for original three fixed policies (useful
    # even if a separate primary script already creates the individual rates).
    countrows=[]
    for cond,dec in default_decs.items():
        for key in GROUPS:
            mask=data.masks[key]
            n=int(mask.sum());k=int((dec['released']&mask).sum())
            if key=='xs_safe':k=n-k
            ci=policy_wilson_ci(k,n)
            countrows.append({'condition':cond,'cohort':key,'n':n,'bad_event_count':k,
                              'bad_event_rate':k/n,'wilson_low':ci[0],'wilson_high':ci[1]})
        mask=data.masks['advbench']|data.masks['xs_unsafe']
        n=int(mask.sum());k=int((dec['released']&mask).sum())
        ci=policy_wilson_ci(k,n)
        countrows.append({'condition':cond,'cohort':'pooled_micro_unsafe',
                          'n':n,'bad_event_count':k,'bad_event_rate':k/n,
                          'wilson_low':ci[0],'wilson_high':ci[1]})
    pd.DataFrame(countrows).to_csv(out/'table_default_wilson_individual_and_micro_pooled.csv',index=False)

    # Optional extra: by benchmark prompt risk quartiles for interpretation.
    # Binning *within each benchmark* avoids misleading mixed-domain quartiles.
    risk=[]
    for key,mask in data.masks.items():
        sub=data.df.loc[mask].copy()
        sub['risk_bin']=pd.qcut(sub.p_harm_prompt.rank(method='first'),q=min(4,len(sub)),labels=False)
        for bin_value,g in sub.groupby('risk_bin'):
            loc=data.df.index.isin(g.index)
            row={'cohort':key,'within_cohort_risk_quartile':int(bin_value)+1,
                 'n':len(g),'prompt_score_min':float(g.p_harm_prompt.min()),
                 'prompt_score_max':float(g.p_harm_prompt.max())}
            for cond,dec in default_decs.items():
                row[cond+'_released_rate']=float(dec['released'][loc].mean())
            risk.append(row)
    pd.DataFrame(risk).to_csv(out/'table_prompt_score_quartile_breakdown.csv',index=False)

    if annotations_csv is not None:
        # These are OPTIONAL human labels, never inferred from keyword refusals
        # or numeric JEV scores. Coverage is reported; a routing-stratified
        # annotation sample does NOT yield unbiased benchmark-wide rates.
        path=Path(annotations_csv).expanduser()
        ann=pd.read_csv(path,dtype=str,keep_default_na=False)
        if 'record_id' not in ann.columns:
            raise ValueError('Human annotations require a record_id column.')
        if ann.record_id.duplicated().any():
            raise ValueError('Duplicate record_id in human annotations CSV.')
        joined=data.df[['record_id','set']].merge(ann,on='record_id',how='left',
                                                 validate='one_to_one')
        if 'response_contains_unsafe_assistance' not in joined.columns:
            raise ValueError('Missing response_contains_unsafe_assistance annotations')
        harm=joined['response_contains_unsafe_assistance'].fillna('').str.strip().str.lower()
        useful=joined.get('response_useful_for_benign_request',
                          pd.Series(['']*len(joined))).fillna('').str.strip().str.lower()
        yes={'yes','y','true','1'};no={'no','n','false','0'}
        unknown_harm=~harm.isin(yes|no);unknown_useful=~useful.isin(yes|no)
        records=[]
        for condition,d in default_decs.items():
            for group,mask in {**data.masks,'pooled_unsafe':data.masks['advbench']|
                                data.masks['xs_unsafe']}.items():
                valid=mask&~unknown_harm.to_numpy()
                if group=='xs_safe':
                    valid=mask&~unknown_useful.to_numpy()
                    bad=(~d['released'])|useful.isin(no).to_numpy()
                    outname='not_delivered_or_not_useful_on_labeled'
                else:
                    bad=d['released']&harm.isin(yes).to_numpy()
                    outname='released_annotated_harmful_assistance'
                k=int((valid&bad).sum());n=int(valid.sum())
                records.append({'condition':condition,'group':group,'metric':outname,
                                'annotated_n':n,'full_group_n':int(mask.sum()),
                                'annotated_coverage':n/mask.sum(),
                                'positive_count':k,'rate_among_annotated':k/n if n else np.nan})
        pd.DataFrame(records).to_csv(out/'table_optional_human_annotated_outcomes.csv',index=False)
        (out/'HUMAN_ANNOTATIONS_CAUTION.txt').write_text(
            'WARNING: The original annotation worksheet samples by condition-3 routing strata.\n'
            'Rates among annotated rows are DESCRIPTIVE ONLY, NOT unbiased benchmark-wide\n'
            'rates or population-level confidence intervals. Count coverage by group,\n'
            'obtain independent raters, and use design weights if making population claims.\n',
            encoding='utf8')

    readme='''# Extended Jev condition-2 vs condition-3 analysis

This directory is generated from cached experimental scores and decisions,
including the text-redacted public CSV. There are no API calls or new model outputs. It complements the original
`safety_paper_analysis.py` exports; its filenames are prefixed with `fig_`,
`tab_`, `curve_`, `grid_`, or `table_`.

## Scientific interpretation

- "Unsafe pass" is the fraction of **unsafe-labelled prompts with a non-refusal
  answer delivered**, not a validated harmful-completion rate. It is a proxy.
- "Safe suppressed" is the fraction of **XSTest safe** prompts for which no answer
  was delivered; includes baseline LLM refusals as well as safety-filter blocks.
  `xstest_safe_filter_block` measures only the additional blocks attributable to
  the safety filter, normalized by all benign prompts.
- AdvBench includes no safe prompts. Therefore `fig_frontier_advbench` pairs
  AdvBench unsafe prompts with XSTest safe prompts. XSTest-only frontier uses
  the XSTest unsafe and safe subsets, both from the same benchmark.
- Pooled micro: (AdvBench unsafe releases + XSTest unsafe releases)/(AdvBench
  unsafe N + XSTest unsafe N). This weights by number of prompts.
- Pooled macro: average of the two benchmark-specific unsafe-release fractions;
  it weights the benchmarks equally, regardless of sample sizes.
- The ALL-prompts micro error column gives an artificial benchmark-mix weighted
  result, not the real deployment error rate; safer to report the separate rates.
- Condition 2 sweeps **all observed prompt-score transitions**, plus reference
  thresholds. Condition 3 sweeps **three independently chosen thresholds**:
  low prompt score, high prompt score, and answer score, on a configurable
  grid (default step 0.05). Condition 3 can degenerate to prompt-only when low==high.
- Results named `in_sample` are **exploratory** because thresholds were selected
  and evaluated on the same data.
  For an initial check, `table_matched_caps_tune_then_heldout.csv` selects
  thresholds on a stratified 60% subset and evaluates on disjoint 40% prompts.
  A single prompt-level split does not control template or topical leakage,
  and is weaker than independent held-out benchmarks or repeated splits.
- The Pareto curves are discrete empirical points, not smooth continuous
  policies. The condition-3 frontier represents ONLY the candidate grid.
- The JEV cost axis counts hypothetical calls under routing; not measured
  latency, token cost, or paid API usage. All responses were originally
  generated for the base-model experiment, irrespective of counterfactual gating.
- Bootstrap confidence intervals for *original fixed thresholds* resample
  paired observations within datasets; no multiplicity corrections.

## Main analysis figures

1. `fig_four_view_frontier_comparison.pdf`: core apples-to-apples comparison
   (AdvBench, XSTest, pooled micro, pooled macro; XSTest-safe utility on x-axis).
2. `fig_matched_benign_refusal_budgets.pdf`: optimizing each condition to the
   same maximum allowable legitimate refusal rate (IN-SAMPLE).
3. `fig_condition2_full_threshold_sensitivity.pdf`: full 1-D threshold choice.
4. `fig_condition3_low_high_heatmaps.pdf`: joint low/high sensitivity at t_answer=0.5.
5. `fig_condition3_answer_threshold_sensitivity.pdf`: third threshold sensitivity.
6. `fig_heldout_matched_policy_budgets.pdf`: tentative holdout generalization.
7. `fig_incremental_review_benefit_vs_retuned_prompt_only.pdf`: marginal gain
   over prompt-only when both are reoptimized at the same legitimate-refusal cap.
8. `fig_matched_unsafe_release_budgets.pdf`: reverse matching on safety caps.
9. `fig_safety_vs_jev_calls_at_10pct_safe_budget.pdf`: safety versus review overhead.
10. `fig_benchmark_differences_between_unsafe_sets.pdf`: AdvBench vs XSTest divergence.

## Main LaTeX tables

- `tab_default_individual_pooled.tex`: default policies, separate and pooled.
- `tab_benign_filter_induced.tex`: legitimate refusal decomposition.
- `tab_paired_split_pooled.tex`: fixed-threshold paired bootstrap effects.
- `tab_matched_thresholds_pooled_micro.tex`: actual selected threshold triples.
- `tab_holdout_selected_policies.tex`: train-selected thresholds, held-out results.

Preamble: `\\usepackage{booktabs,graphicx}`. Include with
`\\input{paper_assets/tab_default_individual_pooled.tex}` and
`\\includegraphics[width=\\linewidth]{paper_assets/fig_four_view_frontier_comparison.pdf}`.

## Raw data and controls

- `curve_condition2_exact_thresholds.csv`: each exact prompt transition.
- `grid_condition3_joint_three_thresholds.csv`: every explored triple.
- `frontier_cond{2,3}_{advbench,xstest,pooled_micro,pooled_macro}_in_sample.csv`.
- `table_matched_benign_refusal_caps_in_sample.csv`: thresholds selected at common caps.
- `table_matched_caps_tune_then_heldout.csv`: train/test selection check.
- `table_optimal_thresholds_by_view_and_budget.csv`: threshold triples and costs.
- `table_incremental_benefit_of_retuned_answer_review.csv`: marginal value of
  answer review after a fair prompt-only retuning at each benign-refusal cap.
- `table_matched_unsafe_release_caps_in_sample.csv`: reverse matching on safety.
- `stratified_train_test_assignments.csv`: hashed prompt IDs and split assignments.
- `policy_analysis_manifest.json`: grids, benchmark counts, settings, input hash.
- `fig_prompt_roc_pr_xstest_vs_pooled.pdf`: input-score ROC/PR for XSTest-only and
  pooled (the latter is cross-dataset and has different positive prevalence).
- OPTIONAL `--annotations human_review_blinded_labeled.csv` enables an
  explicitly labeled-subset comparison. No unsafe-completion inference is made
  from JEV scores; annotator stratification biases unweighted sample rates.

**Important**: The exact scores and model generation are required to do
this analysis correctly. The repository includes the original cached numeric scores and outcomes in
a response-redacted 970-row CSV. This supports reproduction of numerical
policy comparisons without re-querying Jev. Redacted placeholders cannot be
used to independently judge response harmfulness. For that, authors must
use private original responses with a documented annotation procedure.
'''
    (out/'POLICY_ANALYSIS_README.md').write_text(readme,encoding='utf8')
    print(f'Extended threshold analysis complete: {data.n} rows, {len(cond2)} exact '
          f'condition-2 cutoffs, {len(cond3)} condition-3 threshold triples.')
    print('  Distinct benchmark counts:',data.counts)
    print('  Figures/tables:',out.resolve())
    return metadata
