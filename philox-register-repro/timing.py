#!/usr/bin/env python3
"""Summarizes the logs of time.sh.

Usage: timing.py TIMING_DIR BASE_NAME NAME...
For each kernel and executable: the median over the rounds of the per-round median time (ms), and the ratio to
BASE_NAME. The ratio is computed in each round (same round, same GPU temperature); the table gives the median ratio
and, in brackets, the lowest and the highest ratio. Writes TIMING_DIR/summary.tsv and prints a table.
"""
import re
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1])
names = sys.argv[2:]
data, order = {}, []
for n in names:
    for f in sorted(d.glob(f"{n}.r*.log")):
        rnd = int(re.search(r"\.r(\d+)\.log$", f.name).group(1))
        for line in f.read_text().splitlines():
            m = re.match(r"(\S+)\s+k=(\d+)\s+median_ms=\s*([\d.]+).*checksum=(\d+)", line)
            if m:
                key = m.group(1) + (f"(k={m.group(2)})" if m.group(2) != "0" else "")
                if key not in order:
                    order.append(key)
                e = data.setdefault((n, key), {"med": {}, "sum": set()})
                e["med"][rnd] = float(m.group(3))
                e["sum"].add(m.group(4))
rows = []
for key in order:
    base = data[(names[0], key)]["med"]
    for n in names:
        e = data[(n, key)]
        med = list(e["med"].values())
        ratios = [e["med"][r] / base[r] for r in e["med"]]
        rows.append((key, n, len(med), statistics.median(med), statistics.median(ratios), min(ratios), max(ratios),
                     ",".join(sorted(e["sum"]))))
with (d / "summary.tsv").open("w") as out:
    out.write(f"kernel\tname\trounds\tmedian_ms\tratio_to_{names[0]}\tratio_low\tratio_high\tchecksums\n")
    for r in rows:
        out.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.3f}\t{r[4]:.4f}\t{r[5]:.4f}\t{r[6]:.4f}\t{r[7]}\n")
print("kernel".ljust(30) + "".join(f"{n:>30}" for n in names))
for key in order:
    line = key.ljust(30)
    for r in rows:
        if r[0] == key:
            line += f"{r[3]:9.3f} ms {100 * (r[4] - 1):+6.1f}% [{100 * (r[5] - 1):+5.1f},{100 * (r[6] - 1):+5.1f}]"
    print(line)
