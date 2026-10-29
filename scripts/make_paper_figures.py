"""Regenerate the five numbered figures in the October 28, 2026 paper.

Figures 2-5 use public response-redacted scores/outcomes. Figure 1 is a
conceptual diagram. Exact rendering may differ from the manuscript layout,
but every numerical point is calculated from the supplied results.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import Normalize
from jev_repro.safety_paper_analysis import load_results, wilson
from jev_repro.safety_policy_optimization import PolicyData,read_input


def save(fig,folder,name):
    fig.savefig(folder/f'{name}.pdf',bbox_inches='tight',pad_inches=.15)
    fig.savefig(folder/f'{name}.png',bbox_inches='tight',pad_inches=.15,dpi=180)
    plt.close(fig)


def fig1(out):
    fig,ax=plt.subplots(figsize=(10.8,2.5));ax.set_xlim(0,11);ax.set_ylim(0,3);ax.axis('off')
    def box(x,y,w,txt,color='#f2f3f5'):
        ax.add_patch(FancyBboxPatch((x,y),w,.55,boxstyle='round,pad=.04',
                                   fc=color,ec='#7b828b',lw=.8))
        ax.text(x+w/2,y+.275,txt,ha='center',va='center',fontsize=8.5)
    def arrow(x,y,xx,yy):
        ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',
                                    mutation_scale=9,lw=.8,color='#434a54'))
    for i,(name,y) in enumerate([('C1',2.28),('C2',1.34),('C3',.4)]):
        ax.text(.1,y+.28,name,ha='left',va='center',fontweight='bold')
        box(.7,y,1.2,'Prompt');arrow(1.95,y+.27,2.35,y+.27)
        if i==0:
            box(2.4,y,2.0,'Target LLM');arrow(4.45,y+.27,7.65,y+.27)
        elif i==1:
            box(2.4,y,1.75,'Jev input gate','#e9e5f6');arrow(4.2,y+.27,4.55,y+.27)
            box(4.6,y,1.9,'Target LLM');arrow(6.55,y+.27,7.65,y+.27)
        else:
            box(2.4,y,1.75,'Jev input tiers','#e9e5f6');arrow(4.2,y+.27,4.55,y+.27)
            box(4.6,y,1.45,'Target LLM');arrow(6.1,y+.27,6.4,y+.27)
            box(6.45,y,1.55,'Jev output review','#e9e5f6');arrow(8.05,y+.27,8.3,y+.27)
        box(8.4,y,2.2,'Delivered / refused')
    ax.text(5.7,.12,'C3: high tier blocked before generation; low tier bypasses answer review.',
            fontsize=8,ha='center',va='bottom',color='#555555')
    save(fig,out,'fig1_pipeline')


def fig2(df,out):
    fig,ax=plt.subplots(figsize=(8.8,4.6))
    groups=['advbench','xs_unsafe','xs_safe']
    names=['AdvBench\nunsafe answered','XSTest unsafe\nanswered','XSTest safe\nsuppressed']
    conditions=[('cond1','LLM alone','#59677b'),('cond2','Prompt-only Jev','#176AA3'),
                ('cond3','Tiered + review','#d77d35')]
    x=np.arange(3); width=.245
    for j,(col,label,color) in enumerate(conditions):
        rates=[];low=[];high=[]
        for g in groups:
            s=df[df['set']==g];n=len(s)
            k=int((s[col]=='answered').sum()) if g!='xs_safe' else int((s[col]!='answered').sum())
            l,h=wilson(k,n)
            rates.append(100*k/n);low.append(100*(k/n-l));high.append(100*(h-k/n))
        pos=x+(j-1)*width
        ax.bar(pos,rates,width,label=label,color=color)
        ax.errorbar(pos,rates,yerr=[low,high],fmt='none',ecolor='#303030',capsize=3,lw=1)
    ax.set_xticks(x,names);ax.set_ylim(0,101);ax.set_ylabel('Proxy outcome rate (%)')
    ax.set_title('Original reported settings (Wilson 95% intervals)')
    ax.legend(fontsize=8,loc='upper right',ncol=3);ax.grid(axis='y',alpha=.18)
    fig.tight_layout();save(fig,out,'fig2_wilson')


def fig3(df,out):
    p=df.p_harm_prompt.to_numpy(float);baseline=df.cond1.eq('answered').to_numpy(bool)
    masks={k:df['set'].eq(k).to_numpy(bool) for k in ['advbench','xs_unsafe','xs_safe']}
    ts=np.linspace(.1,.9,161)
    fig,ax=plt.subplots(figsize=(6.8,4.2))
    for group,label,color in [('xs_safe','XSTest safe refused','#9c1638'),
                              ('xs_unsafe','XSTest unsafe answered','#205ca6'),
                              ('advbench','AdvBench answered','#6d553c')]:
        vals=[]
        for t in ts:
            release=(p<t)&baseline
            val=1-release[masks[group]].mean() if group=='xs_safe' else release[masks[group]].mean()
            vals.append(val*100)
        ax.plot(ts,vals,label=label,color=color,lw=1.9)
    ax.set(xlabel='Jev input threshold $\\tau$',ylabel='Rate (%)',xlim=(.1,.9))
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();save(fig,out,'fig3_c2_threshold')


def fig4(df,out,asset_dir):
    # Use the exact in-sample Pareto points exported by the standalone optimizer.
    fig,axes=plt.subplots(1,2,figsize=(9.8,3.7),sharex=True)
    policy=PolicyData(read_input(df.attrs['source_path']),seed=42,test_fraction=.4)
    views=[('xstest','xstest_unsafe_pass','(a) XSTest unsafe (200 prompts)'),
           ('pooled_micro','pooled_unsafe_micro_pass','(b) Pooled unsafe (720 prompts)')]
    for ax,(view,metric,title) in zip(axes,views):
        for cond,label,color in [('cond2','C2 optimized frontier','#176AA3'),
                                 ('cond3','C3 optimized frontier','#D17C26')]:
            csv=asset_dir/f'frontier_{cond}_{view}_in_sample.csv'
            data=pd.read_csv(csv)
            ax.plot(data.xstest_safe_refusal*100,data[metric]*100,lw=2,
                    label=label,color=color)
        for cond,t,lo,hi,marker,color,lab in [
            ('cond2',.5,.3,.9,'o','#176AA3','C2 reported'),
            ('cond3',.5,.3,.9,'*','#D17C26','C3 reported'),
            ('cond2',.765,.3,.9,'s','#176AA3','C2 at 4% cap'),
            ('cond3',.45,.60,.80,'D','#D17C26','C3 at 4% cap')]:
            d=policy.decisions(cond,t=t,lo=lo,hi=hi)
            m=policy.aggregate(d)
            ax.scatter(m['xstest_safe_refusal']*100,m[metric]*100,s=75,
                       marker=marker,edgecolors=color,facecolors='white' if marker in ('s','D') else color,
                       linewidths=1.3,zorder=5,label=lab)
        ax.set(title=title,xlabel='XSTest-safe prompts suppressed (%)',xlim=(0,16))
        ax.grid(alpha=.17)
        if view=='xstest':ax.set_ylim(bottom=0,top=27)
        else:ax.set_ylim(bottom=0,top=10)
    axes[0].set_ylabel('Unsafe-labeled prompts answered (%)')
    axes[0].legend(fontsize=6.5,loc='upper right')
    fig.tight_layout();save(fig,out,'fig4_matched_frontiers')


def profiled_f1(df,step=.01):
    """Exact 0.01 threshold grid; profile over answer threshold efficiently.

    For each a, cumulative counts of baseline-answered rows with p_out>=a
    allow all (low,high) configurations to be evaluated without re-running Jev.
    """
    grid=np.round(np.arange(int(round(1/step))+1)*step,10)
    p=df.p_harm_prompt.to_numpy(float)
    q=df.p_harm_answer.to_numpy(float)
    base=df.cond1.eq('answered').to_numpy(bool)
    unsafe=df['set'].ne('xs_safe').to_numpy(bool)
    safe=~unsafe
    n_h=int(unsafe.sum());n_s=int(safe.sum())
    # axis: cutoff index, answer threshold index
    arrays={}
    for name,mask in [('unsafe',unsafe),('safe',safe)]:
        selected=base&mask
        pp=p[selected];qq=q[selected]
        arrays[name]={
            'base':(pp[:,None]<grid[None,:]).sum(axis=0),
            'blocked':np.stack([((pp[:,None]<cut)&(qq[:,None]>=grid[None,:])).sum(axis=0)
                                for cut in grid],axis=0),
        }
    f1=np.full((len(grid),len(grid)),np.nan)
    best_a=np.full((len(grid),len(grid)),np.nan)
    for il,lo in enumerate(grid):
        for ih in range(il,len(grid)):
            u=arrays['unsafe']['base'][ih]-(arrays['unsafe']['blocked'][ih]-arrays['unsafe']['blocked'][il])
            s=arrays['safe']['base'][ih]-(arrays['safe']['blocked'][ih]-arrays['safe']['blocked'][il])
            tp=n_h-u;tn=s;fp=n_s-s;fn=u
            with np.errstate(invalid='ignore',divide='ignore'):
                f1_h=2*tp/(2*tp+fp+fn)
                f1_s=2*tn/(2*tn+fp+fn)
            score=(f1_h+f1_s)/2
            j=int(np.nanargmax(score));f1[ih,il]=score[j];best_a[ih,il]=grid[j]
    return grid,f1,best_a


def fig5(df,out):
    grid,f1,a=profiled_f1(df)
    best=float(np.nanmax(f1))
    candidates=np.argwhere(np.isclose(f1,best,atol=1e-12))
    # Deterministic representative; published point may be tied.
    preferred=(.60,.78)
    il=int(round(preferred[0]*100));ih=int(round(preferred[1]*100))
    if np.isclose(f1[ih,il],best,atol=1e-10):
        chosen=(ih,il)
    else:
        chosen=tuple(candidates[0])
    h,l=chosen
    fig,ax=plt.subplots(figsize=(8.5,6.6))
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#f0f0f0')
    im=ax.imshow(f1,origin='lower',extent=[-.005,1.005,-.005,1.005],
                 aspect='equal',cmap=cmap,vmin=.90,vmax=max(.952,best))
    ax.plot([0,1],[0,1],color='white',lw=1,ls=':')
    ax.scatter([.3],[.9],s=110,facecolors='none',edgecolors='#e34a33',
               linewidths=2,label='Reported C3 ($\\ell=.30$, $h=.90$)')
    ax.scatter([grid[l]],[grid[h]],marker='*',s=170,color='#f1d22a',edgecolors='black',
               linewidths=.4,label='Representative maximum')
    ax.annotate(f'Max = {best:.4f}\n($\\ell={grid[l]:.2f}$, $h={grid[h]:.2f}$, $a={a[h,l]:.2f}$)',
                (grid[l],grid[h]),xytext=(.08,.69),textcoords='axes fraction',
                fontsize=9,weight='bold',arrowprops={'arrowstyle':'->','color':'white'},
                bbox={'facecolor':'white','alpha':.9,'edgecolor':'#aaaaaa'})
    ax.set(xlabel='Lower input threshold $\\ell$',ylabel='Upper input threshold $h$',
           title='Condition 3: best macro-F1 over answer threshold $a$',xlim=(0,1),ylim=(0,1))
    ax.legend(loc='upper left',fontsize=8)
    cbar=fig.colorbar(im,ax=ax,shrink=.85);cbar.set_label('Maximum end-to-end macro-F1')
    fig.tight_layout();save(fig,out,'fig5_c3_profiled_f1')
    pd.DataFrame(f1,index=grid,columns=grid).rename_axis('high').to_csv(out/'fig5_macro_f1_grid.csv')
    pd.DataFrame(a,index=grid,columns=grid).rename_axis('high').to_csv(out/'fig5_best_answer_threshold_grid.csv')
    unsafe=df['set'].ne('xs_safe').to_numpy(bool)
    delivered=df['cond3'].eq('answered').to_numpy(bool)
    tp=int((unsafe&~delivered).sum());tn=int((~unsafe&delivered).sum())
    fp=int((~unsafe&~delivered).sum());fn=int((unsafe&delivered).sum())
    original=(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2
    return {'best_f1':best,'low':float(grid[l]),'high':float(grid[h]),'answer':float(a[h,l]),
            'original_f1':original}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='data/derived/results_public.csv')
    parser.add_argument('--output',default='paper/figures')
    parser.add_argument('--assets',default='paper_assets')
    args=parser.parse_args()
    df=load_results(Path(args.input));df.attrs['source_path']=args.input
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'pdf.fonttype':42,'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    fig1(out);fig2(df,out);fig3(df,out);fig4(df,out,Path(args.assets))
    result=fig5(df,out)
    print('Paper figures 1-5 generated; C3 heatmap best:',result)

if __name__=='__main__':main()
