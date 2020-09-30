import numpy as np
import matplotlib.pyplot as plt

x_dot = 20
Tend = 10
Ts = 0.02
t = np.arange(0, Tend + Ts, Ts)

# Plant block for computing new states with steering angle input
def newe_states(states, U1):
    m = 1500
    Iz = 3000
    Caf = 19000
    Car = 33000
    lf = 2
    lr = 3
    Ts = 0.02
    x_dot = 20
    y_dot = states[0]
    psi = states[1]
    psi_dot = states[2]
    Y = states[3]
    sub_loop = 30  # Chop Ts into 30 pieces
    for k in range(0, sub_loop):
        # Compute the the derivatives of the states
        y_dot_dot = -(2 * Caf + 2 * Car) / (m * x_dot) * y_dot + (-x_dot - (2 * Caf * lf - 2 * Car * lr) / (m * x_dot)) * psi_dot + 2 * Caf / m * U1
        psi_dot = psi_dot
        psi_dot_dot = -(2 * lf * Caf - 2 * lr * Car) / (Iz * x_dot) * y_dot - (2 * lf ** 2 * Caf + 2 * lr ** 2 * Car) / (Iz * x_dot) * psi_dot + 2 * lf * Caf / Iz * U1
        Y_dot = np.sin(psi) * x_dot + np.cos(psi) * y_dot

        # Update the state values with new state derivatives
        y_dot = y_dot + y_dot_dot * Ts / sub_loop
        psi = psi + psi_dot * Ts / sub_loop
        psi_dot = psi_dot + psi_dot_dot * Ts / sub_loop
        Y = Y + Y_dot * Ts / sub_loop

    new_states = states
    new_states[0] = y_dot
    new_states[1] = psi
    new_states[2] = psi_dot
    new_states[3] = Y

    return new_states

# Trajectory formation
x_tr = t * x_dot
y_tr = -9*np.ones(len(x_tr))
dx_tr = x_tr[1:len(x_tr)] - x_tr[0:len(x_tr) - 1]
dy_tr = y_tr[1:len(y_tr)] - y_tr[0:len(y_tr) - 1]
psi = np.zeros(len(x_tr))
psi[0] = np.arctan2(dy_tr[0], dx_tr[0])
psi[1:len(psi)] = np.arctan2(dy_tr[0:len(dy_tr)], dx_tr[0:len(dx_tr)])
dpsi = psi[1:len(psi)] - psi[0:len(psi) - 1]
psint = psi
psint[0] = psi[0]
for i in range(1, len(psint)):
    if dpsi[i - 1] < -np.pi:
        psint[i] = psint[i - 1] + (dpsi[i - 1] + 2 * np.pi)
    elif dpsi[i - 1] > np.pi:
        psint[i] = psint[i - 1] + (dpsi[i - 1] - 2 * np.pi)
    else:
        psint[i] = psint[i - 1] + dpsi[i - 1]

Utotal = np.zeros(len(t))
state_hist = [0, 0, 0, y_tr[0] + 20]
statestotal = np.zeros((len(t), len(state_hist)))
statestotal[0, :] = state_hist

# PID Control Loop
Kp_yaw = 140
Kd_yaw = 60
Ki_yaw = 100
Kp_Y = 7
Kd_Y = 3
Ki_Y = 5
for i in range(0, len(t)-1):
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
    states = newe_states(state_hist, Ut)
    statestotal[(i+1), :] = states

# Comparison of real and reference trajectory
plt.plot(x_tr, statestotal[:, 3], 'r')
plt.plot(x_tr, y_tr, 'b')
plt.show()

# Comparison of real and reference yaw angles
plt.plot(x_tr, psint, 'b')
plt.plot(x_tr, statestotal[:,1], 'r')
plt.show()

# Steering angle values
plt.plot(x_tr, Utotal)
plt.show()
