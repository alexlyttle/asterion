import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'out/glitch_with_err'  # Modify this

n, nu = np.loadtxt(os.path.join(data_dir, 'nu.csv'), delimiter=',', skiprows=1).T
n_pred, nu_pred = np.loadtxt(os.path.join(data_dir, 'nu_pred.csv'), delimiter=',', skiprows=1).T

delta_nu = 111.84  # uHz

fig, ax = plt.subplots()

ax.plot(nu%delta_nu, nu, 'o')
ax.plot(nu_pred%delta_nu, nu_pred)
ax.set_xlabel(f'nu mod. {delta_nu:.2f} uHz')
ax.set_ylabel('nu (uHz)')

plt.show()
