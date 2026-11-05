"""Small allowlisted model registry shared by the local worker and trial UI."""
from pathlib import Path
from .common import BASE, PARENT, EXP, ROOT, read


def models():
    result = [
        {'id': 'current', 'display_name': '旧模型 · 校准监督训练', 'adapter': str(PARENT),
         'kind': 'baseline', 'status': '历史模型，新编辑器对照'},
        {'id': 'base', 'display_name': '原始 Qwen · 未专项训练', 'adapter': None,
         'kind': 'baseline', 'status': '基础模型，新工具提示对照'}]
    selection = read(EXP/'development-selection.json') if (EXP/'development-selection.json').exists() else {}
    for start, name in [('base', '从基础模型训练'), ('current', '从当前模型继续训练')]:
        for seed in (20261104, 20261105):
            run = EXP/'runs'/f'{start}-seed-{seed}'
            path = run/'final'
            if not (path/'adapter_model.safetensors').exists():
                continue
            model_id = f'functional-{start}-{seed}'
            if model_id in selection.get('models', {}):
                path = ROOT/selection['models'][model_id]['adapter']
            result.append({'id': model_id, 'display_name': f'功能训练 · {name} · {seed}',
                           'adapter': str(path), 'kind': 'candidate',
                           'status': '实验模型，AI辅助评审；已知力度、复合要求及对比度存在失败',
                           'default': model_id == selection.get('selected_model_id')})
    round2 = ROOT/'experiments/functional-v2-20261104/development-selection.json'
    if round2.exists():
        for model_id, item in read(round2).get('models', {}).items():
            path=ROOT/item['adapter']
            if not (path/'adapter_model.safetensors').exists():
                continue
            curriculum='curriculum' in model_id
            result.append({'id':model_id, 'display_name':'第二轮 · '+('人像／风景课程 · 第3轮' if curriculum else '旧数据等预算对照 · 第2轮'),
                'adapter':str(path), 'kind':'candidate', 'default':False,
                'status':'实验模型，AI辅助评审；'+('强弱与复合要求改善，仍有英文过曝失败，尚未通过升级门槛' if curriculum else '旧数据等预算继续训练，对照用途，仍有固定力度和漏项')})
    for dirname, label in [('functional-v3-20261104','第三轮'),('functional-v4-20261105','第四轮'),('functional-v6-20261105','第六轮')]:
        record=ROOT/'experiments'/dirname/'development-selection.json'
        if not record.exists():continue
        for model_id,item in read(record).get('models',{}).items():
            path=ROOT/item['adapter']
            if not (path/'adapter_model.safetensors').exists():continue
            round4=dirname.startswith('functional-v4')
            round6=dirname.startswith('functional-v6')
            name=(item.get('display_name') if round4 or round6 else ('偏好优化' if '-dpo-' in model_id else '匹配普通微调'))
            result.append({'id':model_id,'display_name':name if round4 or round6 else label+' · '+name,'adapter':str(path),'kind':'candidate','default':False,
              'status':item.get('trial_status','实验模型，AI辅助评审；纠错未过升级门槛，仍有曝光与力度失败'),
              'runtime_family':'functional_v4' if round4 else 'functional_v1','input_route':item.get('input_route') if round4 else None})
    return result


def resolve(model_id):
    for item in models():
        if item['id'] == model_id:
            return item
    raise ValueError('Unknown or unavailable model: ' + model_id)
