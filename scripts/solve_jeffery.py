import json, os, platform
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad

#Parameters

r, gamma_dot, theta0 = 2.0, 1.0, 0.0
lam = (r**2 - 1.0) / (r**2 + 1.0)
T_axis_th = np.pi * (r + 1.0/r) / abs(gamma_dot)

def rhs(t, theta):
    return [-(gamma_dot/2.0) * (1.0 - lam*np.cos(2.0*theta[0]))]

def rate(th):
    return -(gamma_dot/2.0) * (1.0 - lam*np.cos(2.0*th))

def hit_minus_pi(t, theta):
    return theta[0] + np.pi
hit_minus_pi.terminal, hit_minus_pi.direction = True, -1

t_end = 2.5 * T_axis_th

# run 1 -- event only, gives T_num
ev = solve_ivp(rhs, (0.0, t_end), [theta0], method="RK45",
               rtol=1e-7, atol=1e-9, events=hit_minus_pi)
T_num = ev.t_events[0][0] if len(ev.t_events[0]) else np.nan

# run 2 -- no event, dense sampling, gives the saved trajectory and F02
sol = solve_ivp(rhs, (0.0, t_end), [theta0], method="RK45",
                rtol=1e-7, atol=1e-9, dense_output=True)
t      = np.linspace(0.0, t_end, 2000)
theta  = sol.sol(t)[0]
dtheta = rate(theta)

print(f"lambda            = {lam:.6f}")
print(f"T_axis (theory)   = {T_axis_th:.9f}")
print(f"T_num  (solver)   = {T_num:.9f}")
print(f"relative error    = {abs(T_num-T_axis_th)/T_axis_th*100:.3e} %")
T_quad = 2.0 * quad(lambda th: 1.0/((gamma_dot/2.0)*(1.0-lam*np.cos(2.0*th))), 0.0, np.pi/2.0)[0]
print(f"T_quad            = {T_quad:.9f}")
print(f"|T_num - T_quad|  = {abs(T_num-T_quad):.3e}")
for th in (0.0, -np.pi/2, -np.pi):
    print(f"dtheta/dt at theta={th:+.3f} : {rate(th):+.6f}")

T_tight = solve_ivp(rhs, (0.0, t_end), [theta0], method="RK45",
                    rtol=1e-9, atol=1e-11, events=hit_minus_pi).t_events[0][0]
print(f"T_num (rtol=1e-9)  = {T_tight:.9f}   shift = {abs(T_tight-T_num):.3e}")

os.makedirs("data", exist_ok=True); os.makedirs("figures", exist_ok=True)
np.savetxt("data/case_A.csv", np.column_stack([t, theta, dtheta]),
           delimiter=",", header="t,theta,dtheta_dt", comments="")
with open("data/case_A_settings.json", "w") as f:
    json.dump(dict(case="A", r=r, gdot=gamma_dot, theta0=theta0, lam=lam,
                   method="RK45", rtol=1e-7, atol=1e-9, t_end=t_end, T_num=T_num,
                   python=platform.python_version(), numpy=np.__version__), f, indent=2)

hi, lo = rate(0.0), rate(-np.pi/2)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.10))
a1.plot(t, theta, color="blue", lw=1.6)
a1.axhline(-np.pi, color="0.55", ls="--", lw=0.9)
a1.set_ylabel(r"$\theta$ (rad, unwrapped)", fontsize=10)
a1.set_title("Case A: r = 2, $\\dot{\\gamma}$ = 1, $\\theta(0)=0$", fontsize=15)
a2.plot(t, dtheta, color="red", lw=1.6)
a2.axhline(hi, color="maroon", ls=":", lw=1.9)
a2.axhline(lo, color="black", ls=":", lw=0.9)
a2.set_xlabel("t"); a2.set_ylabel(r"$d\theta/dt$", fontsize=10)
for ax in (a1, a2):
    ax.grid(alpha=0.25)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.10)
fig.savefig("figures/F02_baseline.png", dpi=300, bbox_inches="tight"); plt.close(fig)
