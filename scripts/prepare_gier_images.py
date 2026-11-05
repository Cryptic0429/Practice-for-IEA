import json
from pathlib import Path
from zip_ranges import extract_selected

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT/'metadata/gier-archive-index.json').read_text())
required = set(json.loads((ROOT/'metadata/stage1-required-gier-images.json').read_text()))
selected = [r for r in rows if r['name'] in required]
assert {r['name'] for r in selected} == required
extract_selected(rows, selected, [dict(id='1v9KluqSF_Z4Nb91K0LrobXd0JkqT0z_S',size=7610610367)],
                 ROOT/'.downloads/gier',ROOT/'.downloads/gier-ranges',workers=32,merge_gap=131072)
