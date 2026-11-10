"""Disposable trial worker using exactly the frozen round7 inference contract."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import traceback

from functional_v1.common import ROOT, GPUJob, sha, read
from functional_v1.catalog import resolve
from .diagnose_editor import EXP
from .outcomes import system_prompt, execute, EDITOR_VERSION, SCHEMA_VERSION


def reply(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', choices=['0', '2', '3'], required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit('Owned round7 trial child terminated'))
    spec = resolve(args.model)
    assert spec.get('runtime_family') == 'functional_v7'
    protocol = read(EXP / 'training-protocol.json')
    for name, digest in protocol['code_hashes'].items():
        assert sha(ROOT / name) == digest
    adapter = Path(spec['adapter'])
    assert sha(adapter / 'adapter_model.safetensors') == spec['adapter_sha256']
    with GPUJob(args.gpu, 'interactive-trial-' + spec['id'], max_seconds=7200) as budget:
        with budget.ledger() as value:
            next(j for j in value['jobs'] if j['id'] == budget.id).update(editor=EDITOR_VERSION, schema=SCHEMA_VERSION, runtime='functional_v7.runtime')
        import torch
        from . import runtime
        from functional_v1.editor import load_rgb, save
        torch.set_num_threads(4)
        model, p = runtime.load_model(adapter), runtime.processor()
        identity = dict(
            model_id=spec['id'], display_name=spec['display_name'], adapter_sha256=spec['adapter_sha256'],
            editor=EDITOR_VERSION, schema=SCHEMA_VERSION, runtime_family='functional_v7',
            runtime_sha256=sha(Path(runtime.__file__)), editor_sha256=sha(Path(__file__).with_name('editor_soft.py')),
            outcome_schema_sha256=sha(Path(__file__).with_name('outcomes.py')),
            prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),
            training_data_sha256=protocol['dataset_sha256'], training_route=spec['training_route'])
        reply(dict(status='ready', model_id=spec['id'], identity=identity))
        for line in sys.stdin:
            value = json.loads(line)
            if value.get('command') == 'quit':
                break
            output = None
            try:
                budget.tick()
                image = Path(value['image'])
                output = image.with_name('output.png')
                output.unlink(missing_ok=True)
                case = dict(image=str(image.relative_to(ROOT)), instructions={'original': value['instruction']})
                response, decision = runtime.predict(model, p, case, 'original')
                result = execute(load_rgb(image), decision)
                if result['status'] == 'declined':
                    assert result['result'] is None and not output.exists()
                    reply(dict(status='declined', execution_status='declined', decision=decision,
                               reason_code=result['reason_code'], parameters=None, response=response,
                               output=None, output_sha256=None, identity=identity))
                else:
                    save(output, result['result'])
                    reply(dict(status='executed', execution_status=result['status'], decision=decision,
                               parameters=result['parameters'], response=response, output=str(output),
                               output_sha256=sha(output), identity=identity))
            except Exception as error:
                if output is not None:
                    output.unlink(missing_ok=True)
                traceback.print_exc(file=sys.stderr)
                reply(dict(status='failed', error=f'{type(error).__name__}: {error}', identity=identity))
                if isinstance(error, __import__('functional_v1.common', fromlist=['BudgetExceeded']).BudgetExceeded):
                    break


if __name__ == '__main__':
    main()
