"""Freeze paired seeds/fixed updates before any R22 features or training."""
from .common import ROOT,EXP,DATA,PARENT,PARENT_SHA,read,write,sha,stamp,GPUJob

def main():
    assert not (EXP/'training-protocol.json').exists()
    prior_path=ROOT/'experiments/functional-v21-20261107/training-protocol.json';prior=read(prior_path)
    assert all(sha(ROOT/n)==h for n,h in prior['code_hashes'].items())
    admission=read(DATA/'admission.json');counts=admission['counts'];cases=read(DATA/'cases.json')
    assert counts['new_cases']==54 and counts['excluded_uncertain']==10
    assert len(cases)==979 and len({c['group_id'] for c in cases})==240
    assert counts['quality_refusal_original_groups']==55
    hashes=dict(prior['code_hashes'])
    for path in (ROOT/'scripts/functional_v22').glob('*.py'):hashes[str(path.relative_to(ROOT))]=sha(path)
    sources=dict(prior['sources']);sources['train']=dict(path=str((DATA/'cases.json').relative_to(ROOT)),sha256=sha(DATA/'cases.json'))
    used=GPUJob.consumed(read(ROOT/'experiments/functional-v1-20261104/resources.json'));assert used<23.65
    p=dict(at=stamp(),status='frozen_before_new_GPU_features_or_head_training',hypothesis='Expand explicitly reviewed quality-boundary supervision from23 to55 TRAIN photo groups while keeping the exact R21 parent, head, texture function, loss, optimizer and fixed1160 update budget. Test the same two paired seeds; no lucky-checkpoint choice.',
        prior_protocol=dict(path=str(prior_path.relative_to(ROOT)),sha256=sha(prior_path)),parent=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA,
        sources=sources,sealed_acceptance_sources=prior['sealed_acceptance_sources'],admission_sha256=sha(DATA/'admission.json'),data=counts,
        inherited_hidden_manifests=prior['hidden_manifests'],inherited_hidden_protocol_sha256=prior['hidden_protocol_sha256'],
        inherited_texture=dict(path='experiments/functional-v21-20261107/texture-features.json',sha256=sha(ROOT/'experiments/functional-v21-20261107/texture-features.json')),
        parent_semantics=prior['parent_semantics'],
        controlled_feature_dependency=dict(binding_path='experiments/functional-v21-20261107/controlled-feature-binding.json',binding_sha256=sha(ROOT/'experiments/functional-v21-20261107/controlled-feature-binding.json'),access='Only after each seed passes old/native/control gates. Existing exact input-only cache, not new7B inference.'),
        input_policy=prior['input_policy'],architecture=prior['training']['architecture'],loss=prior['training']['loss'],
        training=dict(seeds=[20261205,20261206],seeds_paired_with_R21=True,updates=1160,save_steps=[1160],
            semantic_batch=16,paired_language_batch=32,learning_rate=.001,weight_decay=.01,CPU_only=True,
            initialization='Independent random head per seed; same seeds as R21, not continued R21 head weights.',
            normalization='Exact frozen R21 normalize; fit TRAIN979 mean/std only, hidden per-inputRMS, std floor.1.',
            sampling='Shuffle979 case IDs with own seeded Torch generator each full pass; contiguous ZH/EN pairs; stop at exactly1160 updates, not20 whole new-data epochs.',
            quality_weight='Same sqrt(edit802/quality76), clipped1..5 rule fit on TRAIN only.',
            matching='Update count, head/parent/texture/loss/optimizer settings matched to R21e20; dataset, normalization/class statistics, sample order and exact presentations necessarily change. Not a full factorial causal proof.'),
        policy='magnitude-quality only at final fixed1160-step weights; risk sigmoid>=.5 unchanged, uncalibrated. Parent actions/active tools/signs preserved; no fallback.',
        evaluation=dict(seeds_both_required=True,old_native_min_completed=4,native_min_completed=16,decision_controls_required=20,controlled_min_completed=81,major_allowed=0,
            order='Each fixed seed: old6 -> native20+controls20 -> controlled100 only if eligible. Any confirmed major failure prevents promotion; do not resolve every other uncertain detail of already failed seeds.',
            locked='Both seeds, final1160updates, same quality policy fixed before training. No development-based checkpoint/seed/threshold substitution.',
            reviews='Anonymous direct Codex visual AI plus native-pixel supplements; exact prior pixel disputes stay uncertain. Decline is not completed editing. Not human/user approval.',
            acceptance='Still sealed; open only after both fixed seeds pass all development gates, then no checkpoint/parameter tuning.',
            later_budget_use='If route fails, no larger training on unverified labels. Remaining GPU allocation may document fixed-model behavior on prospectively frozen additional non-training development/reliability requests; no acceptance relabeling or hidden follow-up training.'),
        resources=dict(total_cap_GPU_hours=24,occupied_at_freeze=used,remaining_GPU_hours=24-used,allowed_GPUs=[0,2,3],GPU1_preserved=True,
            feature_job_max_seconds=900,trial_reserve_GPU_hours=.25,user_authorizes_full24_total=True),
        code_hashes=hashes,L1_used_for_selection=False,independent_user_feedback=0)
    write(EXP/'training-protocol.json',p);print('Frozen R22:979/1958/240, same two seeds and1160updates; noGPU started, remaining',24-used)

if __name__=='__main__':main()
