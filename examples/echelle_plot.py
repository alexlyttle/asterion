import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'docs/source/tutorials/out/glitch_with_err'  # Modify this

n, nu = np.loadtxt(os.path.join(data_dir, 'nu.csv'), delimiter=',', skiprows=1).T
n_pred, nu_pred = np.loadtxt(os.path.join(data_dir, 'nu_pred.csv'), delimiter=',', skiprows=1).T

delta_nu = 111.84  # uHz

fig, ax = plt.subplots()

x = (nu - n*delta_nu)%delta_nu
x_pred = (nu_pred - n_pred*delta_nu)%delta_nu
ax.plot(x, nu, 'o')
ax.plot(x_pred, nu_pred)
ax.set_xlabel(f'nu - {delta_nu:.2f}*n uHz')
ax.set_ylabel('nu (uHz)')

plt.show()
