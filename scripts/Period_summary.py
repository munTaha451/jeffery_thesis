import json
import numpy as np

lines = [["case", "r", "gdot", "T_num", "T_axis", "error_pct"]]
for case in ("A", "B", "C"):
    with open(f"data/case_{case}_settings.json") as f:
        m = json.load(f)
    T_th = np.pi * (m["r"] + 1.0/m["r"]) / abs(m["gdot"])
    err = abs(m["T_num"] - T_th) / T_th * 100.0
    lines.append([case, f"{m['r']:.1f}", f"{m['gdot']:.2f}",
                  f"{m['T_num']:.6f}", f"{T_th:.6f}", f"{err:.2e}"])

with open("data/period_summary.csv", "w") as f:
    f.write(",".join(lines[0]) + "\n")
    for row in lines[1:]:
        f.write(",".join(row) + "\n")
