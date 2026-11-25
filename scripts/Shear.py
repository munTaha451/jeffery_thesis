import numpy as np
import matplotlib.pyplot as plt

#Data
gamma_dot = 1.0
y = np.linspace(-1.0, 1.0)
ux = gamma_dot * y   # linear profile: dux/dy = gdot, uy = 0

#shear flow
G = np.array([[0.0, gamma_dot],
              [0.0, 0.0]])
assert np.allclose(G[0, 1], gamma_dot) and np.allclose(G.sum(axis=1)[1], 0.0)

fig, ax = plt.subplots(figsize=(5.0, 3.4), dpi=300)
ax.plot(y, ux, lw=1.6)
ax.set_xlabel(r"$y$")
ax.set_ylabel(r"$u_x = \dot{\gamma} y$")
ax.set_xlim(-1.0, 1.0)
fig.tight_layout()
fig.savefig("figures/F01_shear_flow.png")

plt.show()


