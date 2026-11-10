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
    round7 = ROOT/'experiments/functional-v7-20261105/development-selection.json'
    if round7.exists():
        for model_id, item in read(round7).get('models', {}).items():
            path = ROOT/item['adapter']
            if not (path/'adapter_model.safetensors').exists():
                continue
            result.append({'id': model_id, 'display_name': item['display_name'],
                           'adapter': str(path), 'adapter_sha256': item['adapter_sha256'],
                           'kind': 'candidate', 'default': False, 'status': item['trial_status'],
                           'runtime_family': 'functional_v7', 'training_route': item['training_route']})
    round9 = ROOT/'experiments/functional-v9-20261106/development-selection.json'
    if round9.exists():
        for model_id, item in read(round9).get('models', {}).items():
            path = ROOT/item['adapter']
            if not (path/'adapter_model.safetensors').exists():
                continue
            result.append({'id': model_id, 'display_name': item['display_name'],
                           'adapter': str(path), 'adapter_sha256': item['adapter_sha256'],
                           'kind': 'candidate', 'default': False, 'status': item['trial_status'],
                           'runtime_family': 'functional_v9', 'training_route': item['training_route'],
                           'origin': item['origin']})
    round10 = ROOT/'experiments/functional-v10-20261106/development-selection.json'
    if round10.exists():
        for model_id, item in read(round10).get('models', {}).items():
            path = ROOT/item['adapter']
            if not (path/'adapter_model.safetensors').exists():
                continue
            result.append({'id': model_id, 'display_name': item['display_name'],
                           'adapter': str(path), 'adapter_sha256': item['adapter_sha256'],
                           'kind': 'candidate', 'default': False, 'status': item['trial_status'],
                           'runtime_family': 'functional_v10', 'training_route': item['training_route'],
                           'origin': 'base'})
    round11 = ROOT/'experiments/functional-v11-20261106/development-selection.json'
    if round11.exists():
        for model_id, item in read(round11).get('models', {}).items():
            path = ROOT/item['adapter']
            if not (path/'adapter_model.safetensors').exists():
                continue
            result.append({'id': model_id, 'display_name': item['display_name'],
                           'adapter': str(path), 'adapter_sha256': item['adapter_sha256'],
                           'kind': 'candidate', 'default': False, 'status': item['trial_status'],
                           'runtime_family': 'functional_v11', 'training_route': item['training_route'],
                           'origin': 'base'})
    round12 = ROOT/'experiments/functional-v12-20261106/development-selection.json'
    if round12.exists():
        for model_id, item in read(round12).get('models', {}).items():
            path = ROOT/item['adapter']
            if not (path/'adapter_model.safetensors').exists():
                continue
            result.append({'id': model_id, 'display_name': item['display_name'],
                           'adapter': str(path), 'adapter_sha256': item['adapter_sha256'],
                           'kind': 'candidate', 'default': False, 'status': item['trial_status'],
                           'runtime_family': 'functional_v12', 'training_route': item['training_route'],
                           'origin': 'base'})
    round13 = ROOT/'experiments/functional-v13-20261106/development-selection.json'
    if round13.exists():
        for model_id, item in read(round13).get('models', {}).items():
            path = ROOT/item['adapter']
            if not (path/'adapter_model.safetensors').exists():
                continue
            result.append({'id': model_id, 'display_name': item['display_name'],
                           'adapter': str(path), 'adapter_sha256': item['adapter_sha256'],
                           'kind': 'candidate', 'default': False, 'status': item['trial_status'],
                           'runtime_family': 'functional_v13', 'training_route': item['training_route'],
                           'origin': 'base'})
    round14 = ROOT/'experiments/functional-v14-20261106/development-selection.json'
    if round14.exists():
        for model_id, item in read(round14).get('models', {}).items():
            path = ROOT/item['adapter']
            if not (path/'adapter_model.safetensors').exists():
                continue
            result.append({'id': model_id, 'display_name': item['display_name'],
                           'adapter': str(path), 'adapter_sha256': item['adapter_sha256'],
                           'kind': 'candidate', 'default': False, 'status': item['trial_status'],
                           'runtime_family': 'functional_v14', 'training_route': item['training_route'],
                           'origin': 'base'})
    round15 = ROOT/'experiments/functional-v15-20261106/development-selection.json'
    if round15.exists():
        for model_id, item in read(round15).get('models', {}).items():
            path = ROOT/item['adapter']
            if not (path/'adapter_model.safetensors').exists():
                continue
            result.append({'id': model_id, 'display_name': item['display_name'],
                           'adapter': str(path), 'adapter_sha256': item['adapter_sha256'],
                           'kind': 'candidate', 'default': False, 'status': item['trial_status'],
                           'runtime_family': 'functional_v15', 'training_route': item['training_route'],
                           'origin': 'base'})
    round18 = ROOT/'experiments/functional-v18-20261107/development-selection.json'
    if round18.exists():
        for model_id,item in read(round18).get('models',{}).items():
            path=ROOT/item['adapter']
            if (path/'adapter_model.safetensors').exists():
                result.append(dict(id=model_id,display_name=item['display_name'],adapter=str(path),adapter_sha256=item['adapter_sha256'],kind='candidate',default=False,status=item['trial_status'],runtime_family='functional_v18',training_route=item['training_route'],origin='base'))
    round19 = ROOT/'experiments/functional-v19-20261107/development-selection.json'
    if round19.exists():
        for model_id,item in read(round19).get('models',{}).items():
            path=ROOT/item['adapter']
            if (path/'adapter_model.safetensors').exists():
                result.append(dict(id=model_id,display_name=item['display_name'],adapter=str(path),adapter_sha256=item['adapter_sha256'],kind='candidate',default=False,status=item['trial_status'],runtime_family='functional_v19',training_route=item['training_route'],origin='base'))
    round20=ROOT/'experiments/functional-v20-20261107/development-selection.json'
    if round20.exists():
        for model_id,item in read(round20).get('models',{}).items():
            path=ROOT/item['controller']
            if path.is_file():
                result.append(dict(id=model_id,display_name=item['display_name'],adapter=str(path),adapter_sha256=item['controller_sha256'],kind='learned_controller',default=False,status=item['trial_status'],runtime_family='functional_v20',training_route=item['route'],origin='base'))
    round21=ROOT/'experiments/functional-v21-20261107/development-selection.json'
    if round21.exists():
        for model_id,item in read(round21).get('models',{}).items():
            path=ROOT/item['controller']
            if path.is_file():
                result.append(dict(id=model_id,display_name=item['display_name'],adapter=str(path),adapter_sha256=item['controller_sha256'],kind='learned_magnitude_quality_head',default=False,status=item['trial_status'],runtime_family='functional_v21',training_route=item['policy'],origin='base'))
    for version in (22,23):
        selection=ROOT/f'experiments/functional-v{version}-20261107/development-selection.json'
        if selection.exists():
            for model_id,item in read(selection).get('models',{}).items():
                path=ROOT/item.get('controller',item.get('adapter',''))
                weight=path if version==22 else path/'adapter_model.safetensors'
                if weight.is_file():
                    result.append(dict(id=model_id,display_name=item['display_name'],adapter=str(path),adapter_sha256=item.get('controller_sha256',item.get('adapter_sha256')),kind='experimental_candidate',default=False,status=item['trial_status'],runtime_family=f'functional_v{version}',training_route=item.get('policy','language_merger'),origin='base'))
    return result


def resolve(model_id):
    for item in models():
        if item['id'] == model_id:
            return item
    raise ValueError('Unknown or unavailable model: ' + model_id)
