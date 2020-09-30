import numpy as np


class SupportFiles:

    def __init__(self):
        g = 9.81
        m = 1500
        Iz = 3000
        Caf = 19000
        Car = 33000
        lf = 2
        lr = 3
        Ts = 0.02
        x_dot = 20

        self.constants = [g, m, Iz, Caf, Car, lf, lr, Ts, x_dot]

    def trajectory_generator(self, t):
        x_dot = self.constants[8]

        x_tr = t * x_dot
        # y_tr = -9*np.ones(len(x_tr))
        # y_tr = 9*np.tanh(t-t[-1]/2)
        y_tr = 9 * np.sin(t)

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

        return psint, x_tr, y_tr

    def newe_states(self, state, u1):
        m = self.constants[1]
        Iz = self.constants[2]
        Caf = self.constants[3]
        Car = self.constants[4]
        lf = self.constants[5]
        lr = self.constants[6]
        T = self.constants[7]
        xdot = self.constants[8]
        y_dot = state[0]
        psai = state[1]
        psi_dot = state[2]
        Y = state[3]
        sub_loop = 30  # Chop Ts into 30 pieces
        for k in range(0, sub_loop):
            # Compute the the derivatives of the states
            y_dot_dot = -(2 * Caf + 2 * Car) / (m * xdot) * y_dot + (
                    -xdot - (2 * Caf * lf - 2 * Car * lr) / (m * xdot)) * psi_dot + 2 * Caf / m * u1
            psi_dot = psi_dot
            psi_dot_dot = -(2 * lf * Caf - 2 * lr * Car) / (Iz * xdot) * y_dot - (
                    2 * lf ** 2 * Caf + 2 * lr ** 2 * Car) / (Iz * xdot) * psi_dot + 2 * lf * Caf / Iz * u1
            Y_dot = np.sin(psai) * xdot + np.cos(psai) * y_dot

            # Update the state values with new state derivatives
            y_dot = y_dot + y_dot_dot * T / sub_loop
            psai = psai + psi_dot * T / sub_loop
            psi_dot = psi_dot + psi_dot_dot * T / sub_loop
            Y = Y + Y_dot * T / sub_loop

        new_states = state
        new_states[0] = y_dot
        new_states[1] = psai
        new_states[2] = psi_dot
        new_states[3] = Y

        return new_states
