"""Record only directly viewed anonymous sheets; preserve all earlier observations."""
import json
from functional_v1.common import ROOT,read,write,append,sha,stamp
EXP=ROOT/'experiments/functional-v21-20261107'


def record(split,bundle,notes,name):
    folder=EXP/'review'/split/bundle;manifest=read(folder/'manifest.json')
    rows=[]
    for cid,judgments in notes.items():
        items=[item for sheet in manifest['sheets'] if sheet['case_id']==cid for item in sheet['items']]
        assert len(items)==len(judgments),(cid,len(items),len(judgments))
        for item,(completion,quality,evidence) in zip(items,judgments):
            assert completion in 'CPFU' and quality in 'AmMU' and len(evidence)>=20
            rows.append(dict(item_id=item,completion=dict(C='completed',P='partial',F='failed',U='uncertain')[completion],
                quality=dict(A='acceptable',m='minor_issue',M='major_issue',U='uncertain')[quality],evidence=evidence))
    path=folder/(name+'.json');assert not path.exists()
    destination=EXP/'review'/split/'codex-observations.jsonl'
    prior={n['item_id'] for n in map(json.loads,destination.read_text().splitlines())} if destination.exists() else set()
    items={i['item_id']:i for i in manifest['items']}
    sheets={item:s for s in manifest['sheets'] for item in s['items']}
    for row in rows:
        assert row['item_id'] not in prior
        sheet=sheets[row['item_id']];assert sha(ROOT/sheet['sheet'])==sheet['sha256']
        assert sha(ROOT/items[row['item_id']]['image'])==items[row['item_id']]['image_sha256']
    write(path,rows)
    for row in rows:
        item=items[row['item_id']];sheet=sheets[row['item_id']]
        append(destination,dict(row,case_id=item['case_id'],image_sha256=item['image_sha256'],
            sheet=sheet['sheet'],sheet_sha256=sheet['sha256'],bundle=bundle,
            reviewer='Codex direct anonymous visual AI review; not human/user feedback',observed_at=stamp()))
    print('Preserved',len(rows),'direct anonymous observations')
