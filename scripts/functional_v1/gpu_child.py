"""A disposable GPU process: exit guarantees model allocations are released."""
import argparse
import json
from pathlib import Path
import signal
import sys
import traceback
from .common import ROOT, GPUJob, sha, system_prompt, EDITOR_VERSION
from .catalog import resolve


def reply(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', choices=['0','2','3'], required=True)
    ap.add_argument('--model', required=True)
    args = ap.parse_args()
    def terminate(*_):
        raise SystemExit('Owned interactive GPU child terminated')
    signal.signal(signal.SIGTERM, terminate)
    spec = resolve(args.model)
    with GPUJob(args.gpu, 'interactive-trial-'+spec['id'], max_seconds=7200) as budget:
        import torch
        from . import runtime as inference_runtime
        if spec.get('runtime_family') == 'functional_v4':
            from functional_v4 import runtime as inference_runtime
            assert spec['input_route'] in ('plain', 'measured')
        from .editor import load_rgb, edit, save
        torch.set_num_threads(4)
        model, p = inference_runtime.load_model(spec['adapter']), inference_runtime.processor()
        reply({'status': 'ready', 'model_id': spec['id']})
        for line in sys.stdin:
            value = json.loads(line)
            if value.get('command') == 'quit':
                break
            try:
                budget.tick()
                image = Path(value['image'])
                case = {'image': str(image.relative_to(ROOT)), 'instructions': {'original': value['instruction']}}
                if spec.get('runtime_family') == 'functional_v4':
                    response, parameters = inference_runtime.predict(model,p,case,'original',spec['input_route'])
                else:
                    response, parameters = inference_runtime.predict(model,p,case,'original')
                output = image.with_name('output.png')
                save(output, edit(load_rgb(image), parameters))
                identity = {'model_id': spec['id'], 'display_name': spec['display_name'],
                            'adapter_sha256': sha(Path(spec['adapter'])/'adapter_model.safetensors') if spec['adapter'] else None,
                            'editor': EDITOR_VERSION, 'runtime_sha256': sha(Path(inference_runtime.__file__)),
                            'editor_sha256': sha(Path(__file__).with_name('editor.py')),
                            'prompt_sha256': __import__('hashlib').sha256((system_prompt()+getattr(inference_runtime,'NOTE','')).encode()).hexdigest()}
                if spec.get('runtime_family') == 'functional_v4':
                    identity.update(input_route=spec['input_route'],runtime_family='functional_v4',feature_version='image-stats-v1' if spec['input_route']=='measured' else None,base_runtime_sha256=sha(Path(__file__).with_name('runtime.py')))
                reply({'status': 'executed', 'parameters': parameters, 'response': response,
                       'output': str(output), 'output_sha256': sha(output), 'identity': identity})
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                reply({'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
                if isinstance(error, __import__('functional_v1.common', fromlist=['BudgetExceeded']).BudgetExceeded):
                    break


if __name__ == '__main__':
    main()
