import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
    return t[i-1] + (level - th[i-1])*(t[i]-t[i-1])/(th[i]-th[i-1])

tA, thA, mA = load("A")
tC, thC, mC = load("C")
tauA, tauC = abs(mA["gdot"])*tA, abs(mC["gdot"])*tC     # scaled time

tA_p = first_crossing(tA, thA, -np.pi)
tC_p = first_crossing(tC, thC, -np.pi)
TA, TC = tA_p*abs(mA["gdot"]), tC_p*abs(mC["gdot"])

thC_on_A = np.interp(tauA, tauC, thC)                   # collapse metric
collapse_max = np.nanmax(np.abs(thC_on_A - thA))
print(f"T_A = {tA_p:.6f}   T_C = {tC_p:.6f}   ratio = {tC_p/tA_p:.4f}")
print(f"tau-periods: A = {TA:.6f}, C = {TC:.6f}")
print(f"max |theta_C(tau) - theta_A(tau)| = {collapse_max:.3e} rad")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.0), dpi=300)

a1.plot(tA, thA, color="red", lw=1.5, label=rf"A: $\dot{{\gamma}}$ = {mA['gdot']:.1f}")
a1.plot(tC, thC, color="green", lw=1.5, label=rf"C: $\dot{{\gamma}}$ = {mC['gdot']:.1f}")
a1.axhline(-np.pi, color="0.55", ls="--", lw=0.9)
for T, lab in ((tA_p, rf"$T_A$ = {tA_p:.2f}"), (tC_p, rf"$T_C$ = {tC_p:.2f}")):
    a1.axvline(T, color="0.65", ls=":", lw=0.8)
    a1.annotate(lab, (T, -np.pi), xytext=(3, 6), textcoords="offset points", fontsize=8.5)
a1.set_xlabel(r"physical time $t$")
a1.set_ylabel(r"$\theta$ (rad, unwrapped)")
a1.set_title("(a) physical time: C takes twice as long", fontsize=10)
a1.set_xlim(0, max(tA[-1], tC[-1]))

a2.plot(tauA, thA, color="red", lw=1.5, label=rf"A: $\tau$ = {mA['gdot']:.1f}$t$")
a2.plot(tauC, thC, color="green", lw=1.5, ls="--", label=rf"C: $\tau$ = {mC['gdot']:.1f}$t$")
a2.axhline(-np.pi, color="0.55", ls="--", lw=0.9)
a2.axvline(TA, color="0.65", ls=":", lw=0.8)
a2.set_xlabel(r"scaled time $\tau = |\dot{\gamma}|\,t$")
a2.set_ylabel(r"$\theta$ (rad, unwrapped)")
a2.set_title(rf"(b) scaled time: curves collapse (max $\Delta\theta$ = {collapse_max:.1e} rad)", fontsize=10)
a2.set_xlim(0, tauC[-1])

for ax in (a1, a2):
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.15, wspace=0.22)
fig.savefig("figures/F04_shear_effect.png", bbox_inches="tight")
