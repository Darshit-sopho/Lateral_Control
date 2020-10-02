import numpy as np
import matplotlib.pyplot as plt
import Supporting as Spp

support = Spp.SupportFiles()
constants = support.constants
x_dot = constants[8]
Tend = 10
Ts = constants[7]
t = np.arange(0, Tend + Ts, Ts)

psint, x_tr, y_tr = support.trajectory_generator(t)
Utotal = np.zeros(len(t))
Ut = 0
Utotal[0] = Ut
state_hist = [0, psint[0], 0, y_tr[0]+20]
statestotal = np.zeros((len(t), len(state_hist)))
statestotal[0, :] = state_hist
y_err = np.zeros(len(y_tr))
old_states = state_hist
refsignal = np.zeros((len(y_tr)*2, 1))
outputs = 2
k = 0
for i in range(0, refsignal.shape[0], outputs):
    refsignal[i] = psint[k]
    refsignal[i+1] = y_tr[k]
    k = k + 1

Q = np.array([[7000, 0], [0, 115]])
S = np.array([[7000, 0], [0, 115]])
R = np.array([[10000]])
Hz = 20
Ad, Bd, Cd = support.state_matrices()

####################################################################################################
# MPC loop
####################################################################################################

k = 0
for i in range(len(t) - 1):
    state_hist = statestotal[i][0:len(state_hist)]
    X_aug = np.concatenate((state_hist, [Ut]), axis=0)
    X_aug = X_aug.reshape((len(X_aug), 1))
    k = k + outputs
    if k + outputs * Hz <= len(refsignal):
        r = refsignal[k:k + outputs * Hz]
    else:
        r = refsignal[k:len(refsignal)]
        Hz = Hz - 1

    Hr, Fr = support.final_matrices(Ad, Bd, Cd, S, Q, R, Hz)
    last = np.concatenate((X_aug, r), axis=0)
    first = np.matmul(-(np.linalg.inv(Hr)), np.transpose(Fr))
    delu = np.matmul(first, last)

    Ut = Ut + delu[0][0]
    if Ut < -np.pi / 6:
        Ut = -np.pi / 6
    elif Ut > np.pi / 6:
        Ut = np.pi / 6
    else:
        Ut = Ut

    Utotal[i + 1] = Ut
    states = support.newe_states(state_hist, Ut)
    statestotal[(i + 1), :] = states

####################################################################################################
# Plots
####################################################################################################

# Comparison of real and reference trajectory
plt.plot(x_tr, statestotal[:, 3], 'r', linewidth='2', label='Car trajectory')
plt.plot(x_tr, y_tr, '--b', linewidth='2', label='Reference trajectory')
plt.xlabel('x-position [m]')
plt.ylabel('y-position [m]')
plt.legend(fontsize='small', loc='best')
plt.show()

# Comparison of real and reference yaw angles
plt.plot(x_tr, psint, '--b', linewidth='2', label='Reference yaw angles')
plt.plot(x_tr, statestotal[:, 1], 'r', linewidth='2', label='Car yaw angles')
plt.xlabel('x-position [m]')
plt.ylabel('yaw angles [rad]')
plt.legend(fontsize='small', loc='best')
plt.show()

# Steering angle values
plt.plot(x_tr, Utotal * 180 / np.pi, label='Steering wheel angle')
plt.xlabel('x-position [m]')
plt.ylabel('steering wheel angle [degrees]')
plt.legend(fontsize='small', loc='best')
plt.show()

# Error in y position
plt.plot(x_tr, y_err, label='y-error')
plt.xlabel('x-position [m]')
plt.ylabel('y-error [rad]')
plt.legend(fontsize='small', loc='best')
plt.show()