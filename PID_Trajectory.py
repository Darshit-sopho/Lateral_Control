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
state_hist = [0, psint[0], 0, y_tr[0]]
statestotal = np.zeros((len(t), len(state_hist)))
statestotal[0, :] = state_hist
y_err = np.zeros(len(y_tr))
old_states = state_hist

# PID Control Loop
Kp_yaw = 2.5
Kd_yaw = 0
Ki_yaw = 2.5
Kp_Y = 2.5
Kd_Y = 0
Ki_Y = 2.5
for i in range(0, len(t) - 1):
    Ut = Utotal[i]
    if i == 0:
        e_int_y = 0
        e_int_yaw = 0
    else:
        state_hist = statestotal[i][0:len(state_hist)]
        e_y = y_tr[i - 1] - old_states[3]
        e_yf = y_tr[i] - state_hist[3]
        e_dot_y = (e_yf - e_y) / Ts
        e_int_y = e_y + (e_yf + e_y) * Ts / 2
        U_y = Kp_Y * e_yf + Kd_Y * e_dot_y + Ki_Y * e_int_y
        y_err[i] = e_yf
        e_yaw = psint[i - 1] - old_states[1]
        e_yawf = psint[i] - state_hist[1]
        e_dot_yaw = (e_yawf - e_yaw) / Ts
        e_int_yaw = e_yaw + (e_yawf + e_yaw) * Ts / 2
        U_yaw = Kp_yaw * e_yawf + Kd_yaw * e_dot_yaw + Ki_yaw * e_int_yaw

        Ut = U_y + U_yaw
    old_states = state_hist
    if Ut < -np.pi / 6:
        Ut = -np.pi / 6
    elif Ut > np.pi / 6:
        Ut = np.pi / 6
    else:
        Ut = Ut

    Utotal[i + 1] = Ut
    states = support.newe_states(state_hist, Ut)
    statestotal[(i + 1), :] = states

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
