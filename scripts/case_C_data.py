# code/solve_case_C.py
"""Case C: r = 2, gdot = 0.5. Only the shear rate differs from Case A."""
import json, os, platform
import numpy as np
from scipy.integrate import solve_ivp

r, gamma_dot, theta0 = 2.0, 0.5, 0.0
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

ev = solve_ivp(rhs, (0.0, t_end), [theta0], rtol=1e-7, atol=1e-9, events=hit_minus_pi)
T_num = ev.t_events[0][0] if len(ev.t_events[0]) else np.nan

sol = solve_ivp(rhs, (0.0, t_end), [theta0], rtol=1e-7, atol=1e-9, dense_output=True)
t      = np.linspace(0.0, t_end, 2000)
theta  = sol.sol(t)[0]
dtheta = rate(theta)

T_tight = solve_ivp(rhs, (0.0, t_end), [theta0], rtol=1e-9, atol=1e-11,
                    events=hit_minus_pi).t_events[0][0]

print(f"lambda            = {lam:.6f}")
print(f"T_axis (theory)   = {T_axis_th:.9f}")
print(f"T_num  (solver)   = {T_num:.9f}")
print(f"relative error    = {abs(T_num-T_axis_th)/T_axis_th*100:.3e} %")
print(f"T_num (rtol=1e-9) = {T_tight:.9f}  shift = {abs(T_tight-T_num):.3e}")
for th in (0.0, -np.pi/2):
    print(f"dtheta/dt at theta={th:+.3f} : {rate(th):+.6f}")

os.makedirs("data", exist_ok=True)
np.savetxt("data/case_C.csv", np.column_stack([t, theta, dtheta]),
           delimiter=",", header="t,theta,dtheta_dt", comments="")
with open("data/case_C_settings.json", "w") as f:
    json.dump(dict(case="C", r=r, gdot=gamma_dot, theta0=theta0, lam=lam,
                   method="RK45", rtol=1e-7, atol=1e-9, t_end=t_end, T_num=T_num,
                   python=platform.python_version(), numpy=np.__version__), f, indent=2)
