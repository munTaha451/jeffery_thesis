import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

GAMMA_dot = 1.0                  # shear rate, all cases
R_CASES = (2.0, 3.0, 5.0)        # 3.0 is the optional intermediate case
N_PERIODS = 60                   # several complete cycles
BINS = 60
ALPHA = np.deg2rad(15.0)         # +/-15 deg around the flow direction


#Equations
def lam_of(r):
    return (r**2 - 1.0) / (r**2 + 1.0)

def T_axis(r):
    return np.pi * (r + 1.0/r) / abs(GAMMA_dot)

def p_theory(theta, r):
    lam = lam_of(r)
    return np.sqrt(1.0 - lam**2) / (np.pi * (1.0 - lam*np.cos(2.0*theta)))

def S_theory(r):
    return (r - 1.0) / (r + 1.0)

def P_theory(r, alpha=ALPHA):
    return (2.0/np.pi) * np.arctan(r * np.tan(alpha))

def fold(theta):
    return (theta + 0.5*np.pi) % np.pi - 0.5*np.pi


#solve
def trajectory(r, n_periods=N_PERIODS, dt=None):
    lam = lam_of(r)
    T = T_axis(r)
    dt = T/1200.0 if dt is None else dt

    def rhs(t, y):
        return [-(GAMMA_dot/2.0) * (1.0 - lam*np.cos(2.0*y[0]))]

    sol = solve_ivp(rhs, (0.0, n_periods*T), [0.0], method="RK45",
                    rtol=1e-9, atol=1e-11, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)

    t = np.arange(0.0, n_periods*T, dt)      # uniform in physical time
    return t, sol.sol(t)[0]


def statistics(r, n_periods=N_PERIODS, dt=None):
    t, theta = trajectory(r, n_periods, dt)
    th = fold(theta)

    S = float(np.mean(np.cos(2.0*th)))
    P = float(np.mean(np.abs(th) <= ALPHA))

    hist, edges = np.histogram(th, bins=BINS, range=(-np.pi/2, np.pi/2), density=True)
    centres = 0.5*(edges[1:] + edges[:-1])
    p_ref = p_theory(centres, r)
    dtheta = edges[1] - edges[0]

    return dict(r=r, lam=lam_of(r), S=S, S_th=S_theory(r), P=P, P_th=P_theory(r),
                S_err=abs(S - S_theory(r)), P_err=abs(P - P_theory(r)),
                norm=float(np.sum(p_ref)*dtheta),
                l1=float(np.sum(np.abs(hist - p_ref))*dtheta),
                cdf_dev=float(np.max(np.abs(np.cumsum(hist)*dtheta
                                            - np.cumsum(p_ref)*dtheta))),
                hist=hist, centres=centres, p_ref=p_ref, th=th, t=t, dt=dtheta,
                n=th.size)


#main function
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    res = {r: statistics(r) for r in R_CASES}

    print(f"{'r':>4} {'lam':>7} {'S_num':>8} {'S_th':>7} {'|dS|':>7} "
          f"{'P_num':>7} {'P_th':>7} {'|dP|':>7} {'int p':>7} {'L1':>7} {'maxCDF':>8}")
    for r, d in res.items():
        print(f"{r:>4.1f} {d['lam']:>7.4f} {d['S']:>8.4f} {d['S_th']:>7.4f} "
              f"{d['S_err']:>8.1e} {d['P']:>7.4f} {d['P_th']:>7.4f} "
              f"{d['P_err']:>8.1e} {d['norm']:>7.4f} {d['l1']:>7.4f} {d['cdf_dev']:>7.4f}")

# sensitivity to the uniform sampling interval (halved)
    print("\nsensitivity to sampling interval:")
    for r in R_CASES:
        fine = statistics(r, dt=res[r]['dt']/2.0)
        print(f"  r={r:.0f}: dS={abs(fine['S']-res[r]['S']):.2e}  "
              f"dP={abs(fine['P']-res[r]['P']):.2e}  "
              f"dCDF={abs(fine['cdf_dev']-res[r]['cdf_dev']):.2e}")

#data file
    rows = [("r", "lambda", "S_num", "S_theory", "dS",
             "P15_num", "P15_theory", "dP15", "samples")]
    for r, d in res.items():
        rows.append((f"{r:.1f}", f"{d['lam']:.6f}", f"{d['S']:.6f}", f"{d['S_th']:.6f}",
                     f"{d['S_err']:.2e}", f"{d['P']:.6f}", f"{d['P_th']:.6f}",
                     f"{d['P_err']:.2e}", str(d['n'])))
    with open("data/P(15)_statistics.csv", "w") as f:
        for row in rows:
            f.write(",".join(row) + "\n")

#Figure05: numerical vs analytical density
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), dpi=300)
    for ax, r in zip(axes, (2.0, 5.0)):
        d = res[r]
        ax.bar(d["centres"], d["hist"], width=d["dt"], color="grey",
               edgecolor="none", alpha=0.85, label="numerical (uniform time histogram)")
        grid = np.linspace(-np.pi/2, np.pi/2, 600)
        ax.plot(grid, p_theory(grid, r), color="red", lw=1.6,
                label="analytical $p(\\theta)$")
        ax.set_xlim(-np.pi/2, np.pi/2)
        ax.set_xlabel(r"orientation angle $\theta$ (rad)")
        ax.set_ylabel(r"probability density $p(\theta)$")
        ax.set_title(f"r = {r:.0f}   ($\\lambda$ = {d['lam']:.3f})", fontsize=10)
        ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
        ax.set_xticklabels([r"$-\pi/2$", r"$-\pi/4$", "0", r"$\pi/4$", r"$\pi/2$"])
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ax.grid(alpha=0.25)

    fig.suptitle("Orientation density over one axis period", y=1.02)
    fig.savefig("figures/F05_orientation_density.png", bbox_inches="tight")
    plt.close(fig)

#Figure06: S and P15 against aspect ratio
    rr = np.linspace(1.02, 8.0, 400)
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=300)
    ax.plot(rr, [S_theory(x) for x in rr], color="black", lw=1.7, label="$S$ (theory)")
    ax.plot(rr, [P_theory(x) for x in rr], color="purple", lw=1.7, ls="--",
            label="P15 (theory)")
    ax.plot([d["r"] for d in res.values()], [d["S"] for d in res.values()], "o",
            ms=6, color="black", label="numerical $S$")
    ax.plot([d["r"] for d in res.values()], [d["P"] for d in res.values()], "o",
            ms=6, mfc="none", mew=1.4, color="purple", label="P15 (numerical)")
    ax.set_xlabel("aspect ratio  $r$")
    ax.set_ylabel("order parameter $S$")
    ax.set_xlim(1, 8)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    fig.suptitle("Time-averaged orientation statistics vs aspect ratio", y=1.02)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.14)
    fig.savefig("figures/F06_S_vsP(15).png", bbox_inches="tight")
    plt.close(fig)
