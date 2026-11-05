"""Inspect the entire old training pool without changing it or training a model."""
import collections
import json
import re
import sys
import time
import cv2
import numpy as np
from .common import ROOT, EXP, read, write, sha, stamp, TOOLS
from .editor import load_rgb, edit, remap_old_label, statistics


LOCAL = re.compile(r'remove|erase|cut out|background|only.*(?:face|dress|shirt)|replace|插座|移除', re.I)
GENERIC = re.compile(r'^\s*(?:please\s*)?(?:improve|enhance|edit|fix|make.*better|optimi[sz]e)(?:\s+(?:this|the|my|image|photo|picture))*[.!\s]*$', re.I)


def main():
    source = ROOT / 'data/training/quality-v2-20261103/train3k.json'
    rows = read(source)
    out = EXP / 'audit'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'old-training-replay.jsonl'
    if path.exists():
        old = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r['id'] for r in old] == [r['id'] for r in rows[:len(old)]]
    else:
        old = []
    sys.path.insert(0, str(ROOT / 'repo'))
    from src.image_editor import ImageEditor
    cv2.setNumThreads(2)
    started = time.monotonic()
    for row in rows[len(old):]:
        instruction = row['extra_info']['instruction'].strip()
        item = {'id': row['id'], 'group_id': row['group_id'], 'instruction': instruction,
                'empty_instruction': not instruction,
                'generic_instruction_proxy': bool(GENERIC.fullmatch(instruction)),
                'possibly_local_or_out_of_scope_proxy': bool(LOCAL.search(instruction)),
                'unsupported_label_tools': sorted(set(row['extra_info']['label']) - set(TOOLS))}
        try:
            rgb = load_rgb(row['images'][0])
            image = ImageEditor(row['images'][0], resize=False).process_image(row['extra_info']['label'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            revised = edit(rgb, remap_old_label(row['extra_info']['label']))
            item.update(executable=True, input_statistics=statistics(rgb),
                        old_label_statistics=statistics(image), new_label_statistics=statistics(revised),
                        image_shape=list(rgb.shape),
                        old_parameters=row['extra_info']['label'],
                        remapped_parameters=remap_old_label(row['extra_info']['label']))
        except Exception as error:
            item.update(executable=False, error=f'{type(error).__name__}: {error}')
        with path.open('a') as f:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        old.append(item)
        if len(old) % 25 == 0 or len(old) == len(rows):
            write(out / 'status.json', {'status': 'running' if len(old) < len(rows) else 'completed',
                                      'pid': __import__('os').getpid(), 'completed': len(old),
                                      'total': len(rows), 'elapsed_seconds': time.monotonic()-started})
    report = {'status': 'completed', 'completed_at': stamp(), 'source': str(source.relative_to(ROOT)),
              'source_sha256': sha(source), 'rows': len(rows),
              'source_groups': len({r['group_id'] for r in rows}),
              'empty_instructions': sum(r['empty_instruction'] for r in old),
              'generic_instruction_proxy': sum(r['generic_instruction_proxy'] for r in old),
              'possibly_local_or_out_of_scope_proxy': sum(r['possibly_local_or_out_of_scope_proxy'] for r in old),
              'executable_labels': sum(r['executable'] for r in old),
              'unsupported_label_tools': dict(collections.Counter(k for r in old for k in r['unsupported_label_tools'])),
              'limits': ['Full program replay and mechanical screening, not a semantic acceptance of all 3000 labels.',
                         'Local/generic request detection is a screening proxy, not a final semantic classification.',
                         'Rewritten requests require new target checks; original references are not automatically applicable.']}
    write(out / 'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
