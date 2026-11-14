#!/usr/bin/env python3
"""Read frozen local evidence; write derived report tables and a provenance snapshot.

No model inference, training, server access, or edits to experiment evidence.
"""
from pathlib import Path
import ast
import hashlib
import json
import re
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GENERATED = HERE / "generated"
SOURCES = {}


def read(path):
    p = ROOT / path
    raw = p.read_bytes()
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def textread(path):
    p = ROOT / path
    raw = p.read_bytes()
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8", errors="replace")


def esc(value):
    return str(value).translate(str.maketrans({"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}))


def row(*cells):
    return " & ".join(esc(c) for c in cells) + r" \\" + "\n"


def expdir(n):
    hits = list((ROOT / "experiments").glob(f"functional-v{n}-*"))
    assert len(hits) == 1, (n, hits)
    return hits[0]


# Stable report IDs; each entry is one catalogued adapter, never an ancestor stack.
SPECS = [
    ("R1-B", 1, "from-base/sft-functional/seed-01", "基础起点 e3", "functional-base-20261104", 330),
    ("R1-C", 1, "from-sft-calibrated/sft-functional/seed-01", "继续训练 e1", "functional-current-20261104", 110),
    ("R1-R", 1, "from-sft-calibrated/sft-functional/seed-02", "继续训练复核 e1", "functional-current-20261105", 110),
    ("R2-C", 2, "from-functional-v1/sft-curriculum/seed-01", "课程 SFT e3", "curriculum-e3", 540),
    ("R2-S", 2, "from-functional-v1/sft-control/seed-01", "旧数据对照 e2", "control-e2", 360),
    ("R3-D", 3, "from-functional-v2/dpo-correction/seed-01", "DPO 纠错 e1", "dpo-e1", 24),
    ("R3-S", 3, "from-functional-v2/sft-correction/seed-01", "SFT 纠错 e1", "sft-e1", 24),
    ("R4-P", 4, "from-functional-v2/sft-input-plain/seed-01", "普通输入 e1", "plain-e1", 230),
    ("R4-M", 4, "from-functional-v2/sft-input-measured/seed-01", "统计观测输入 e1", "measured-e1", 230),
    ("R6-B", 6, "from-base/sft-native-coverage/seed-01", "基础起点 e4", "base-e4", 920),
    ("R6-C", 6, "from-functional-v2/sft-native-coverage/seed-01", "继续训练 e4", "continued-e4", 920),
    ("R7-L", 7, "from-functional-v6/sft-outcome-language/seed-01", "语言层 e4", "language-e4", 924),
    ("R7-M", 7, "from-functional-v6/sft-outcome-language-merger/seed-01", "语言及连接层 e4", "language-merger-e4", 924),
    ("R9-B", 9, "from-base/sft-fresh-outcome/seed-01", "Qwen 起点 e4", "base-e4", 924),
    ("R9-A", 9, "from-author-iea/sft-fresh-outcome/seed-01", "作者起点 e4", "author-e4", 924),
    ("R10-L", 10, "from-functional-v9/sft-condition-language/seed-01", "语言层 e2", "language-e2", 240),
    ("R10-M", 10, "from-functional-v9/sft-condition-language-merger/seed-01", "语言及连接层 e2", "language-merger-e2", 240),
    ("R11", 11, "from-functional-v10/sft-negative-gain-correction/seed-01", "负曝光纠错 e1", "language-merger-e1", 336),
    ("R12", 12, "from-functional-v11/sft-manual-calibration/seed-01", "工具说明微调 e1", "language-merger-e1", 125),
    ("R13-N", 13, "from-functional-v11/sft-real-coverage/language_merger/seed-01", "原微调范围 e1", "language-merger-e1", 463),
    ("R13-W", 13, "from-functional-v11/sft-real-coverage/all_language_merger/seed-01", "扩展语言范围 e1", "all-language-merger-e1", 463),
    ("R14", 14, "from-functional-v11/sft-visual-late4/seed-01", "视觉末四层 e1", "visual-late4-e1", 463),
    ("R15", 15, "from-functional-v11/sft-shadow-correction/seed-01", "阴影纠错 e1", "language-merger-e1", 463),
    ("R18-P", 18, "from-functional-v15/sft-counterfactual-ranking/seed-01", "偏好排序 e2", "counterfactual-rank-e2", 1002),
    ("R18-S", 18, "from-functional-v15/sft-repeated-positive-control/seed-01", "匹配 SFT e2", "matched-sft-e2", 1002),
    ("R19", 19, "from-functional-v15/sft-photometric-bridge/seed-01", "图像统计分支 e2", "photometric-bridge-e2", 1002),
    ("R23-1", 23, "from-functional-v15/sft-quality-coverage/seed-20261207", "种子20261207", "lora-seed-20261207-step250", 250),
    ("R23-2", 23, "from-functional-v15/sft-quality-coverage/seed-20261208", "种子20261208", "lora-seed-20261208-step250", 250),
]


def summary(n, split, model):
    matches = list((expdir(n) / "evaluation").glob(f"*/{split}/{model}/direct-functional-summary.json"))
    if not matches:
        return None
    assert len(matches) == 1, matches
    d = read(matches[0].relative_to(ROOT))
    return {"completed": d["fully_completed_both_languages"],
            "major": d.get("major_quality_issues"),
            "path": str(matches[0].relative_to(ROOT)),
            "cases_sha256": d.get("cases_sha256")}


def training_logs(n, modelpath, step, sid):
    meta_path = ROOT / "models" / modelpath / "metadata.json"
    meta = read(meta_path.relative_to(ROOT)) if meta_path.exists() else {}
    cp = meta.get("checkpoint", "")
    records = []
    source = None
    if n == 23 and not cp:
        seed = "20261207" if sid == "R23-1" else "20261208"
        cp = str((expdir(n) / "runs" / ("seed-"+seed) / "checkpoint-250").relative_to(ROOT))
    if cp and isinstance(cp, str):
        run = (ROOT / cp).parent
        for p in [run / "trainer_state.json", run / "metrics.jsonl"]:
            if p.exists():
                source = str(p.relative_to(ROOT))
                if p.name == "trainer_state.json":
                    records = read(source).get("log_history", [])
                else:
                    records = [json.loads(x) for x in textread(source).splitlines()]
                break
    fallbacks = {"R1-B": "train-base-20261104.log", "R1-C": "train-current-20261104.log",
                 "R6-B": "train-base.log", "R6-C": "train-continued.log"}
    if not records and sid in fallbacks:
        p = expdir(n) / fallbacks[sid]
        source = str(p.relative_to(ROOT))
        text = textread(source)
        for candidate in re.findall(r"\{[^{}\r\n]+\}", text):
            if "'loss'" not in candidate:
                continue
            try:
                d = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
            d["step"] = len(records) + 1
            records.append(d)
    selected = [x for x in records if "loss" in x and 0 < x.get("step", 0) <= step]
    if selected:
        assert len(selected) == step, (sid, len(selected), step)
    return {"mean_logged_loss": mean(x["loss"] for x in selected) if selected else None,
            "logged_steps": len(selected), "selected_step": step, "source": source,
            "objective": "DPO+SFT" if sid == "R3-D" else "SFT+ranking" if sid == "R18-P" else "SFT"}


def fmt(result, denominator):
    return "--" if result is None else f"{result['completed']}/{denominator}"


def main():
    GENERATED.mkdir(parents=True, exist_ok=True)
    catalog = read("models/catalog.json")
    unique = {x["canonical_path"]: x for x in catalog["models"]}
    heads = {p:m for p,m in unique.items() if "controller" in p}
    adapters = {p:m for p,m in unique.items() if p not in heads}
    assert len(adapters) == 35 and len(heads) == 6
    pixel = read("metadata/goal-v2-20261103/final-analysis-summary.json")
    early_names = {"A":"标准 SFT", "B":"重采样 SFT", "SFT":"继续 SFT", "GRPO":"L1 奖励 GRPO",
                   "Seed20261103":"校准 SFT / 种子03", "Seed20261104":"校准 SFT / 种子04", "Seed20261105":"校准 SFT / 种子05"}
    early = sorted(pixel["models"].items(), key=lambda x:x[1]["mean_l1"])
    table = ""
    for i, (key, d) in enumerate(early, 1):
        gain = (1-d["mean_l1"]/pixel["models"]["A"]["mean_l1"])*100
        table += row(i, early_names[key], f"{d['mean_l1']:.6f}", f"{gain:+.2f}%", f"{d['executable']}/71")
    table += row("参照", "NoEdit（保留原图）", f"{pixel['NoEdit_l1']:.6f}", "不适用", "不适用")
    (GENERATED / "pixel_rows.tex").write_text(table)
    models = []
    for sid,n,path,label,evalid,step in SPECS:
        assert "models/"+path in adapters
        item = dict(id=sid, round=n, path="models/"+path, label=label, evaluation_id=evalid)
        item["training"] = training_logs(n,path,step,sid)
        if n == 1:
            item["acceptance"] = summary(n,"acceptance",evalid)
        else:
            item["native"] = summary(n,"native-development",evalid)
            item["controlled"] = summary(n,"development",evalid)
            item["old"] = summary(n,"old-native",evalid) or summary(n,"natural-use",evalid)
        models.append(item)
    # Every catalogue adapter must appear either in the early table or functional inventory.
    early_paths = [m["canonical_path"] for m in catalog["models"][:7]]
    assert set(early_paths + [m["path"] for m in models]) == set(adapters)
    (GENERATED / "r1_rows.tex").write_text("".join(row(m["id"],m["label"],fmt(m["acceptance"],50)) for m in models[:3]))
    native_rows = ""
    for m in models[3:]:
        a,b = m["native"],m["controlled"]
        majors = "/".join("--" if x is None or x["major"] is None else str(x["major"]) for x in [a,b])
        native_rows += row(m["id"], m["label"], fmt(a,20),fmt(m["old"],6),fmt(b,100),majors)
    (GENERATED / "functional_rows.tex").write_text(native_rows)
    loss_order = sorted((m for m in models if m["training"]["objective"]=="SFT" and m["training"]["mean_logged_loss"] is not None),key=lambda m:m["training"]["mean_logged_loss"])
    loss_rows=""
    for i,m in enumerate(loss_order,1):
        loss_rows += row(i,m["id"]+" "+m["label"],m["training"]["selected_step"],f"{m['training']['mean_logged_loss']:.5f}")
    for m in models:
        if m["training"]["mean_logged_loss"] is None:
            loss_rows += row("--",m["id"]+" "+m["label"],m["training"]["selected_step"],"未取得日志")
    (GENERATED / "loss_rows.tex").write_text(loss_rows)
    early_ids = ["A","B","SFT","GRPO","CAL-03","CAL-04","CAL-05"]
    inventory = list(zip(early_ids,early_paths))+[(m["id"],m["path"]) for m in models]
    inv=""
    for sid,path in inventory:
        inv += esc(sid)+" & "+r"\path{"+path+"}"+r" \\"+"\n"
    (GENERATED / "inventory_rows.tex").write_text(inv)
    table_names = {"PixelRows": "pixel_rows.tex", "RoneRows": "r1_rows.tex", "FunctionalRows": "functional_rows.tex", "LossRows": "loss_rows.tex", "InventoryRows": "inventory_rows.tex"}
    macro_text = ""
    for name, filename in table_names.items():
        macro_text += "\\newcommand{\\" + name + "}{%\n" + (GENERATED / filename).read_text() + "}\n"
    (GENERATED / "tables.tex").write_text(macro_text)
    for filename in table_names.values():
        (GENERATED / filename).unlink()
    citations=[("模型登记", "models/catalog.json"),
               ("早期独立71图分析", "metadata/goal-v2-20261103/final-analysis-summary.json"),
               ("历史实验与校准训练", "实验记录与结论.md")]
    for n in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]:
        p=expdir(n)/"REPORT.md"
        if p.exists():citations.append((f"R{n}报告",str(p.relative_to(ROOT))))
    citations += [("交互台验证", "docs/COMPARISON_STUDIO.md"), ("交互台运行", "scripts/comparison_studio/README.md"),
                  ("R15冻结训练协议", "experiments/functional-v15-20261106/training-protocol.json"),
                  ("R21控制头及损失实现", "scripts/functional_v21/head.py")]
    sources_tex=""
    for i,(label,path) in enumerate(citations,1):
        textread(path)
        display_path = (r"\path{"+path+"}") if path.isascii() else (r"\textnormal{"+esc(path)+"}")
        sources_tex += r"\item[E"+str(i).zfill(2)+"] "+esc(label)+r"\quad"+display_path+"\n"
    (GENERATED/"sources.tex").write_text(sources_tex)
    snapshot={"as_of":"2026-11-12 Asia/Shanghai","catalog_entries":len(catalog['models']),"unique_adapters":35,"unique_registered_heads":6,
              "pixel_test":pixel,"functional_adapters":models,"registered_paths":sorted(unique),"source_sha256":SOURCES,
              "loss_definition":"Arithmetic mean of per-optimizer-step rounded logged losses from this run start through exported checkpoint; no inference performed.",
              "missing_training_logs":[m['id'] for m in models if m['training']['mean_logged_loss'] is None],
              "scope":"Historical evidence snapshot; not a new benchmark or live resource/feedback audit."}
    (GENERATED/"evidence_snapshot.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"adapters":35,"heads":6,"functional":len(models),"loss_available":sum(m['training']['mean_logged_loss'] is not None for m in models),"source_files":len(SOURCES)},ensure_ascii=False))


if __name__ == "__main__":
    main()
