"""Use the same direct review functions in an explicit replication namespace."""
import argparse
from .common import EXP,read


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['record','finalize','native'])
    ap.add_argument('--split',required=True,choices=['old-native','native-development','development'])
    ap.add_argument('--name',required=True);ap.add_argument('--notes');args=ap.parse_args();rep=EXP/'replication'
    if args.command=='record':
        from . import record_notes as helper
        helper.EXP=rep;helper.record(args.split,'locked-replicate',read(rep/'review'/args.split/'locked-replicate'/args.notes),args.name)
    elif args.command=='native':
        from . import native_review as helper
        helper.EXP=rep;helper.prepare(args.split,'locked-replicate',native_folder=args.name)
    else:
        from . import finalize_review as helper
        from . import visual_review as visual
        from functional_v6 import visual_review as established
        helper.EXP=rep;helper.EVAL=rep/'evaluation'/visual.EVAL.name
        def configure():
            visual.configure();established.EXP=rep;established.EVAL=helper.EVAL
        helper.configure=configure;helper.finalize(args.split,['magnitude-quality-e20'],args.name)


if __name__=='__main__':main()
