"""Normal-exposure style requests prevent making every edit a normalization task."""
from functional_v1.common import ROOT,read,write,sha,stamp
from functional_v1.editor import load_rgb,save
from .prepare_counterfactuals import EXP,render,INSTRUCTIONS
from .extend_candidates import panel,COMPOUND
from .prepare_darkening import REQUEST as DARK_REQUEST


def main():
    destination=EXP/'style-candidates/manifest.json';assert not destination.exists()
    source=EXP/'counterfactual-candidates/manifest.json';dark=read(EXP/'darkening-candidates/manifest.json')
    dark_lookup={c['id']:c for c in dark['cases']};cases=[]
    for c in read(source)['cases']:
        cid=c['id'];base=load_rgb(ROOT/c['images']['canonical']['path'])
        bright_path=ROOT/dark_lookup[cid]['images']['bright-shallow-input']['path']
        bright=load_rgb(bright_path)
        settings={'normal-light':dict(exposure=40),
            'normal-compound':dict(exposure=10,contrast=20,shadows=15,saturation=15),
            'normal-dark-style':dict(exposure=-60),'bright-dark-style':dict(exposure=-65)}
        images={}
        for name,p in settings.items():
            result=render(bright if name=='bright-dark-style' else base,p)
            file=EXP/'style-candidates/images'/f'{cid}-{name}.png';save(file,result)
            images[name]=dict(path=str(file.relative_to(ROOT)),sha256=sha(file))
        paths={name:ROOT/r['path'] for name,r in images.items()}
        sheet=panel(cid,'normal-and-dark-style',dict(canonical=ROOT/c['images']['canonical']['path'],
            **{'normal-light':paths['normal-light'],'normal-compound':paths['normal-compound'],
               'normal-dark-style':paths['normal-dark-style'],'bright-input':bright_path,'bright-dark-style':paths['bright-dark-style']}))
        cases.append(dict(id=cid,domain=c['domain'],source_id=c['source_id'],group_id=c['group_id'],images=images,parameters=settings,
            panel=sheet,requests=dict(light=INSTRUCTIONS,compound=COMPOUND,dark=DARK_REQUEST),
            input_roles={'normal-light':'canonical','normal-compound':'canonical','normal-dark-style':'canonical','bright-dark-style':'bright-shallow-input'},
            status='CPU_candidates_only_unreviewed_not_admitted'))
    write(destination,dict(at=stamp(),status='CPU_style_candidates_no_admission_no_training',source_manifest_sha256=sha(source),
        cases=cases,purpose='Same normal input receives genuinely opposite light/dark style requests. Strong darkening is not taught only as recovery of a synthetically overexposed photograph.',
        tentative_final_conditions=dict(light_and_compound=['deep-clean-input','canonical','mild-noise-input','noisy-input'],
            dark=['canonical','bright-shallow-input']),
        older_shallow_restoration_candidates='Preserved in previous manifests, not automatically adopted. Final selection requires visible quality and strength review.',
        proposed_new_semantic_cases=200,proposed_anchor_semantic_cases=40,proposed_total_bilingual_rows=480,
        all_actual_label_decisions_pending=True))
    print('Added normal-input light/compound/dark styles and bright-input strong darkening; no labels admitted.')

if __name__=='__main__':main()
