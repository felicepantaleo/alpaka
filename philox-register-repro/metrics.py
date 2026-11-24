#!/usr/bin/env python3
"""Prints the resource usage of every kernel of every build in a result directory.

Usage: metrics.py RESULT_DIR
CUDA builds: RESULT_DIR/<bench>/<tree>/sm<NN>/{ptxas.log,resusage.txt,sass.txt}
HIP builds:  RESULT_DIR/<bench>-hip/<tree>/<gfx>/compile.log
Writes RESULT_DIR/metrics-cuda.tsv and RESULT_DIR/metrics-hip.tsv and prints both.
CUDA columns: registers, stack frame, spill stores and loads (ptxas -v); STACK and LOCAL (cuobjdump -res-usage);
SASS instructions, LDL, STL and branch instructions (cuobjdump -sass).
HIP columns: VGPRs, SGPRs, ScratchSize, VGPR and SGPR spills, occupancy (-Rpass-analysis=kernel-resource-usage).
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
kernel_re = re.compile(r"gpuKernelI\d+([KR]\d+_[A-Za-z0-9]+?)NS_")

cuda = []
for d in sorted(p for p in root.glob("*/*/sm*") if (p / "ptxas.log").exists()):
    bench, tree, target = d.parent.parent.name, d.parent.name, d.name
    m = {}
    log = (d / "ptxas.log").read_text().splitlines()
    cur = None
    for i, line in enumerate(log):
        if "Function properties for" in line:
            k = kernel_re.search(line)
            cur = k.group(1) if k else None
            if cur:
                s = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log[i + 1])
                m[cur] = dict(frame=s.group(1), sstore=s.group(2), sload=s.group(3))
        elif cur and "Used" in line and "registers" in line:
            m[cur]["reg"] = re.search(r"Used (\d+) registers", line).group(1)
            cur = None
    res = (d / "resusage.txt").read_text().splitlines()
    for i, line in enumerate(res):
        k = kernel_re.search(line)
        if k and line.strip().startswith("Function") and k.group(1) in m:
            m[k.group(1)].update(stack=re.search(r"STACK:(\d+)", res[i + 1]).group(1),
                                 local=re.search(r"LOCAL:(\d+)", res[i + 1]).group(1))
    cur = None
    for line in (d / "sass.txt").read_text().splitlines():
        if "Function :" in line:
            k = kernel_re.search(line)
            cur = k.group(1) if k and k.group(1) in m else None
            if cur:
                m[cur].update(ins=0, ldl=0, stl=0, bra=0)
            continue
        ins = re.match(r"\s+/\*[0-9a-f]{4,}\*/\s+(@!?U?P\d\s+)?([A-Z0-9_.]+)\b", line)
        if cur and ins:
            op = ins.group(2).split(".")[0]
            c = m[cur]
            c["ins"] += 1
            c["ldl"] += op == "LDL"
            c["stl"] += op == "STL"
            c["bra"] += op in ("BRA", "BRX", "JMP", "JMX")
    for k in sorted(m):
        c = m[k]
        cuda.append([bench, tree, target, k] + [str(c[x]) for x in ("reg", "frame", "sstore", "sload", "stack", "local", "ins", "ldl", "stl", "bra")])

hip = []
fields = (("VGPRs", "vgpr"), ("TotalSGPRs", "sgpr"), ("ScratchSize [bytes/lane]", "scratch"), ("VGPRs Spill", "vspill"),
          ("SGPRs Spill", "sspill"), ("Occupancy [waves/SIMD]", "occ"))
for d in sorted(p for p in root.glob("*-hip/*/*") if (p / "compile.log").exists()):
    bench, tree, target = d.parent.parent.name[: -len("-hip")], d.parent.name, d.name
    m = {}
    cur = None
    for line in (d / "compile.log").read_text().splitlines():
        if "remark: Function Name:" in line:
            k = kernel_re.search(line)
            cur = k.group(1) if k else None
            if cur:
                m[cur] = {}
            continue
        if cur:
            for label, key in fields:
                f = re.search(r"remark:\s+" + re.escape(label) + r": (\S+) \[", line)
                if f:
                    m[cur][key] = f.group(1)
    for k in sorted(m):
        hip.append([bench, tree, target, k] + [m[k].get(key, "?") for _, key in fields])

head_c = ["bench", "tree", "target", "kernel", "registers", "stack_frame", "spill_stores", "spill_loads", "STACK", "LOCAL", "sass_instructions", "LDL", "STL", "branches"]
head_h = ["bench", "tree", "target", "kernel", "VGPRs", "SGPRs", "ScratchSize", "VGPR_spill", "SGPR_spill", "occupancy"]
for name, head, rows in (("metrics-cuda.tsv", head_c, cuda), ("metrics-hip.tsv", head_h, hip)):
    text = "\t".join(head) + "\n" + "".join("\t".join(r) + "\n" for r in rows)
    (root / name).write_text(text)
    print(f"== {name}: {len(rows)} rows")
    print(text)
