import json
import numpy as np
import matplotlib.pyplot as plt

def load(case):
    d = np.genfromtxt(f"data/case_{case}.csv", delimiter=",", names=True)
    with open(f"data/case_{case}_settings.json") as f:
        m = json.load(f)
    return d["t"], d["theta"], m

def first_crossing(t, th, level):
    i = int(np.argmax(th <= level))
    if i == 0:
        return np.nan
    return t[i-1] + (level - th[i-1])*(t[i] - t[i-1])/(th[i] - th[i-1])

tA, thA, mA = load("A")
tB, thB, mB = load("B")
TA, TB = first_crossing(tA, thA, -np.pi), first_crossing(tB, thB, -np.pi)
print(f"T_A = {TA:.6f}   T_B = {TB:.6f}   T_B/T_A = {TB/TA:.4f}")

fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=300)
ax.plot(tA, thA, color="red", lw=1.5,
        label=rf"A: $r$ = {mA['r']:.0f}, $\lambda$ = {mA['lam']:.2f}")
ax.plot(tB, thB, color="cyan", lw=1.5,
        label=rf"B: $r$ = {mB['r']:.0f}, $\lambda$ = {mB['lam']:.3f}")
ax.axhline(-np.pi, color="0.55", ls="--", lw=0.9)
for T, lab, dy in ((TA, rf"$T_A$ = {TA:.2f}", 16), (TB, rf"$T_B$ = {TB:.2f}", 6)):
    ax.axvline(T, color="green", ls=":", lw=0.8)
    ax.annotate(lab, (T, -np.pi), xytext=(-4, dy), textcoords="offset points",
                ha="right", fontsize=9)
ax.plot([TA, TB], [-np.pi, -np.pi], "o", ms=4, color="0.2", zorder=5)
ax.set_xlabel(r"time $t$ (dimensionless, $\dot{\gamma}=1$)")
ax.set_ylabel(r"orientation angle $\theta$ (rad, unwrapped)")
ax.set_xlim(0, min(tA[-1], tB[-1]))
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.grid(alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.11, right=0.97, top=0.96, bottom=0.14)
fig.savefig("figures/F03_shape_effect.png", bbox_inches="tight")
