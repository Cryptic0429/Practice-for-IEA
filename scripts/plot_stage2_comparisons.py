"""Paired full-cohort L1 effects from completed, frozen Stage2 test predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT=Path('/mnt/sdc/zhc/IEA')


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--output',type=Path,required=True);args=cli.parse_args()
    source=ROOT/'outputs/stage2-A-edit-grpo-v1/summary.json'
    summary=json.loads(source.read_text());assert summary['status']=='complete'
    for path,digest in summary['prediction_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    labels=['GRPO vs A','GRPO vs SFT','GRPO vs NoEdit','SFT vs A','SFT vs NoEdit']
    keys=['GRPO_vs_A','GRPO_vs_SFT','GRPO_vs_NoEdit','SFT_vs_A','SFT_vs_NoEdit']
    rows=[]
    for split,n in [('test_new',100),('test_regression',200)]:
        result=summary['results'][split]
        assert all(m['rows']==n for m in result['metrics'].values())
        for label,key in zip(labels,keys):
            effect=result['comparisons'][key]
            rows.append(dict(split=split,n=n,comparison=label,mean=effect['mean'],
                ci_low=effect['ci95'][0],ci_high=effect['ci95'][1]))
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'paired-l1-source.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':7,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7})
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,104/25.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.20,right=.97,bottom=.29,top=.86,wspace=.28)
    lo=min(0.,min(r['ci_low'] for r in rows));hi=max(0.,max(r['ci_high'] for r in rows))
    margin=max(.001,(hi-lo)*.12)
    for ax,split,title,letter in zip(axes,['test_new','test_regression'],
        ['New held-out groups (n = 100)','Previously inspected regression (n = 200)'],'ab'):
        selected=[r for r in rows if r['split']==split]
        for y,row in enumerate(selected):
            color='#376A91' if row['comparison'].startswith('GRPO') else '#777777'
            ax.hlines(y,row['ci_low'],row['ci_high'],color=color,lw=1.1)
            ax.plot(row['mean'],y,'o',color=color,ms=4)
        ax.axvline(0,color='#777777',ls='--',lw=.65)
        ax.set_yticks(range(5),labels=labels);ax.set_ylim(4.6,-.6);ax.set_xlim(lo-margin,hi+margin)
        ax.set_xticks([-.03,-.02,-.01,0.,.01],labels=['−0.03','−0.02','−0.01','0','0.01'])
        ax.set_xlabel('Mean L1 improvement (positive is better)',labelpad=8)
        ax.set_title(title,pad=12)
        ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-28,9),textcoords='offset points',
                    fontweight='bold',fontsize=8,ha='left',va='bottom')
    fig.text(.20,.08,'Dots: paired mean improvement; lines: 95% percentile bootstrap intervals.\n'
             '10,000 original-image-group resamples; one training seed; no multiplicity correction.\n'
             'All rows retained, including invalid-output fallback. Image L1 is not human editing preference.',
             fontsize=6,va='bottom',linespacing=1.4)
    base=args.output/'paired-image-quality'
    require_matplotlib_panel_alignment(fig,json_out=str(base)+'.alignment.json',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    fig.savefig(str(base)+'.pdf');fig.savefig(str(base)+'.svg');fig.savefig(str(base)+'.png',dpi=600)
    plt.close(fig)
    audits={}
    for tool,flags in [('audit_pdf_text.py',['--min-pt','5']),('audit_figure_collisions.py',[])]:
        run=subprocess.run([sys.executable,str(ROOT/'scripts'/tool),str(base)+'.pdf',*flags,'--json'],capture_output=True,text=True)
        try: report=json.loads(run.stdout)
        except json.JSONDecodeError: report=dict(stdout=run.stdout,stderr=run.stderr)
        audits[tool]=dict(returncode=run.returncode,report=report)
    report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plotted_comparisons=len(rows),excluded_rows=0,audits=audits,direct_visual_review_required=True)
    (args.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(a['returncode']==0 for a in audits.values()),'Rendered QA requires repair before delivery'
    print(json.dumps(dict(output=str(args.output),direct_visual_review_required=True)))


if __name__=='__main__':
    main()
