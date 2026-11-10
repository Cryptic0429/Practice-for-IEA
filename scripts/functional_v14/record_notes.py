"""Record only directly viewed anonymous sheets; preserve all earlier observations."""
from functional_v1.common import read,write
from .runtime import EXP
from .visual_review import configure
from functional_v6 import visual_review as established


def record(split,bundle,notes,name):
    configure();folder=EXP/'review'/split/bundle;manifest=read(folder/'manifest.json')
    rows=[]
    for cid,judgments in notes.items():
        items=[item for sheet in manifest['sheets'] if sheet['case_id']==cid for item in sheet['items']]
        assert len(items)==len(judgments),(cid,len(items),len(judgments))
        for item,(completion,quality,evidence) in zip(items,judgments):
            assert completion in 'CPFU' and quality in 'AmMU' and len(evidence)>=20
            rows.append(dict(item_id=item,completion=dict(C='completed',P='partial',F='failed',U='uncertain')[completion],
                quality=dict(A='acceptable',m='minor_issue',M='major_issue',U='uncertain')[quality],evidence=evidence))
    path=folder/(name+'.json');assert not path.exists();write(path,rows);established.record(folder,path)
    print('Preserved',len(rows),'direct anonymous observations')
