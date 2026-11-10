"""Compact writer for explicit direct observations, never auto-grades images."""
import json
from pathlib import Path
from .record_primary import record


def record_pairs(notes):
 root=Path(__file__).resolve().parents[2];exp=root/'experiments/functional-v13-20261106'
 bank={i['id']:i for i in json.loads((exp/'paired-candidates/manifest.json').read_text())['items']}
 rows=[]
 for sid,negative,positive in notes:
  for direction,note in [('negative',negative),('positive',positive)]:
   completion,quality,evidence=note
   rows.append((sid+'-paired-'+direction,'executed',completion,quality,evidence,
    'The other panel result has its own opposite request and separate observation; neither is a competing candidate for this instruction.'))
  keep=sid+'-keep'
  if keep in bank:
   rows.append((keep,'executed','C','A','The unchanged input photograph is directly visible on this same inspected panel. The separately executed keep result is verified pixel-identical to that input; no edit or restoration is requested or claimed.',
    'Keep uses exact identity; paired edited requests have independent labels and must not be substituted for this result.'))
 record(rows,bundle='paired-candidates',observations='paired-observations.jsonl')
