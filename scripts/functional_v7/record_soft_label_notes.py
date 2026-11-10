"""Record a directly inspected panel without revealing its old/new role mapping.

This is a note writer, not an image reviewer or label admission script. Runs on
the local machine without image or model dependencies.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def record(number, notes):
    root=Path(__file__).resolve().parents[2]
    folder=root/'experiments/functional-v7-20261105/soft-label-candidates'
    manifest=json.loads((folder/'manifest.json').read_text())
    sheet=manifest['sheets'][number-1]
    expected={v['label']:r['case_id'] for r in sheet['entries'] for v in r['variants']}
    assert set(notes)==set(expected), 'Notes must cover exactly the visible labels in this panel'
    image=root/sheet['sheet']
    assert hashlib.sha256(image.read_bytes()).hexdigest()==sheet['sha256']
    target=folder/'direct-label-observations.json'
    data=json.loads(target.read_text()) if target.exists() else dict(
        reviewer='Codex direct visual AI review; not human',
        role_mapping_hidden_during_panel_observation=True,
        policy='No automatic admission; inspect material details at full resolution. Exact same-pixel review disagreements remain uncertain.',
        observations=[])
    assert not any(r['sheet']==sheet['sheet'] for r in data['observations']), 'Preserve earlier notes; do not overwrite a review'
    for label,(completion,quality,evidence) in notes.items():
        assert completion in ('C','P','F','U') and quality in ('A','m','M','U') and evidence.strip()
        data['observations'].append(dict(
            at=dt.datetime.now(dt.timezone.utc).isoformat(),sheet=sheet['sheet'],sheet_sha256=sheet['sha256'],
            case_id=expected[label],label=label,
            completion={'C':'completed','P':'partial','F':'failed','U':'uncertain'}[completion],
            quality={'A':'acceptable','m':'minor_issue','M':'major_issue','U':'uncertain'}[quality],
            evidence=evidence,inspection='full-frame panel; no pixel-detail claim beyond visible scale',
            admitted_for_new_training=False))
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temp.replace(target)
    print(f'Recorded panel {number}; total {len(data["observations"])} visible result observations. No labels admitted.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('number',type=int);ap.add_argument('notes_json')
    args=ap.parse_args();record(args.number,json.loads(args.notes_json))
