"""Plot complete formal GRPO optimization and validation traces from raw evidence."""
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


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--output',type=Path,required=True);args=cli.parse_args()
    run=ROOT/'outputs/stage2-A-edit-grpo-v1/formal-v1'
    verification=json.loads((run/'training-verification.json').read_text())
    assert verification['step']==128 and verification['candidates']==2048
    candidates=[json.loads(x) for x in (run/'candidates.jsonl').read_text().splitlines()]
    logs=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    metrics={r['step']:r for r in logs if 'grad_norm' in r}
    assert set(metrics)==set(range(1,129))
    selection=json.loads((run/'validation-selection.json').read_text())
    validation=sorted(selection['validations'],key=lambda x:x['step'])
    assert [r['step'] for r in validation]==[64,128]
    rows=[]
    for step in range(1,129):
        batch=[r for r in candidates if r['step']==step]
        assert len(batch)==16 and len({r['id'] for r in batch})==4
        groups=[[r for r in batch if r['id']==sample] for sample in sorted({r['id'] for r in batch})]
        rows.append(dict(step=step,mean_reward=sum(r['reward'] for r in batch)/16,
            kl=metrics[step]['kl'],grad_norm=metrics[step]['grad_norm'],
            valid_fraction=sum(r['parsing']['valid'] and r['scoring']['editor_success'] for r in batch)/16,
            informative_group_fraction=sum(len({r['reward'] for r in g})>1 for g in groups)/4,
            mean_output_tokens=sum(r['output_tokens'] for r in batch)/16))
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'training-source.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (args.output/'validation-source.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['step','mean_l1']);writer.writeheader()
        writer.writerows({k:r[k] for k in ['step','mean_l1']} for r in validation)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'legend.frameon':False})
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,132/25.4))
    fig.subplots_adjust(left=.12,right=.96,bottom=.22,top=.91,hspace=.65,wspace=.40)
    blue='#376A91';orange='#B66B34';gray='#777777'
    steps=[r['step'] for r in rows]
    for ax,label,title in zip(axes.flat,'abcd',['Candidate rewards','Policy drift','Usable learning signal','Validation selection']):
        ax.annotate(label,xy=(0,1),xycoords='axes fraction',xytext=(-29,7),textcoords='offset points',
                    fontweight='bold',fontsize=8,ha='left',va='bottom')
        ax.set_title(title,pad=10)
        ax.set_xlabel('Optimizer step')
    axes[0,0].plot(steps,[r['mean_reward'] for r in rows],color=blue,lw=.9)
    axes[0,0].axhline(0,color=gray,lw=.5,ls='--');axes[0,0].set_ylabel('Mean training reward')
    axes[0,1].plot(steps,[r['kl'] for r in rows],color=blue,lw=.9);axes[0,1].set_ylabel('Mean KL estimate')
    axes[1,0].plot(steps,[r['valid_fraction'] for r in rows],color=blue,lw=.9,label='Executable candidates')
    axes[1,0].plot(steps,[r['informative_group_fraction'] for r in rows],color=orange,lw=.9,ls='--',label='Distinct-reward groups')
    axes[1,0].set_ylabel('Fraction');axes[1,0].set_ylim(-.04,1.04)
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.085),ncol=2,fontsize=6)
    axes[1,1].plot([r['step'] for r in validation],[r['mean_l1'] for r in validation],
                   color=blue,lw=1,marker='o',ms=4)
    axes[1,1].set_xticks([64,128]);axes[1,1].set_xlim(56,136);axes[1,1].set_ylabel('Mean image L1 (lower is better)')
    fig.text(.12,.035,'One seed; raw step means without smoothing. Each update: 4 prompts, 16 candidates.\n'
             'Validation: 200 fixed images. These traces do not establish held-out image-quality gains.',fontsize=6,va='bottom')
    base=args.output/'training-and-selection'
    require_matplotlib_panel_alignment(fig,json_out=str(base)+'.alignment.json',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    fig.savefig(str(base)+'.pdf')
    fig.savefig(str(base)+'.svg')
    fig.savefig(str(base)+'.png',dpi=600)
    plt.close(fig)
    audits={}
    for tool,parameters in [('audit_pdf_text.py',['--min-pt','5']),('audit_figure_collisions.py',[])]:
        result=subprocess.run([sys.executable,str(ROOT/'scripts'/tool),str(base)+'.pdf',*parameters,'--json'],
                              capture_output=True,text=True)
        try: report=json.loads(result.stdout)
        except json.JSONDecodeError: report=dict(stdout=result.stdout,stderr=result.stderr)
        audits[tool]=dict(returncode=result.returncode,report=report)
    files=[run/'metrics.jsonl',run/'candidates.jsonl',run/'validation-selection.json',Path(__file__)]
    report=dict(source_sha256={str(p):sha(p) for p in files},plotted_steps=128,validation_steps=[64,128],
        excluded_records='Only final aggregate train_metrics log is omitted; all128 optimizer records and2048 candidates are retained.',
        uncertainty='Single-run optimization diagnostics, no seed uncertainty interval; validation is the fixed200-image selection cohort.',
        audits=audits,direct_visual_review_required=True)
    (args.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(a['returncode']==0 for a in audits.values()),'Rendered QA needs repair before delivery'
    print(json.dumps(dict(output=str(args.output),direct_visual_review_required=True)))


if __name__=='__main__':
    main()
