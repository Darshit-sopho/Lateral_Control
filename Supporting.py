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
        y_tr = -9*np.ones(len(x_tr))
        # y_tr = 9*np.tanh(t-t[-1]/2)
        # y_tr = 9 * np.sin(t)

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
            y_dot_dot = -(2 * Caf + 2 * Car) / (m * xdot) * y_dot + (-xdot - (2 * Caf * lf - 2 * Car * lr) / (m * xdot)) * psi_dot + 2 * Caf / m * u1
            psi_dot = psi_dot
            psi_dot_dot = -(2 * lf * Caf - 2 * lr * Car) / (Iz * xdot) * y_dot - (2 * lf ** 2 * Caf + 2 * lr ** 2 * Car) / (Iz * xdot) * psi_dot + 2 * lf * Caf / Iz * u1
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

    def state_matrices(self):
        m = self.constants[1]
        Iz = self.constants[2]
        Caf = self.constants[3]
        Car = self.constants[4]
        lf = self.constants[5]
        lr = self.constants[6]
        T = self.constants[7]
        xdot = self.constants[8]
        a = -(2*Caf + 2*Car)/(m*xdot)
        b = (-xdot - (2 * Caf * lf - 2 * Car * lr) / (m * xdot))
        c = -(2 * lf * Caf - 2 * lr * Car) / (Iz * xdot)
        d = - (2 * lf ** 2 * Caf + 2 * lr ** 2 * Car) / (Iz * xdot)
        e = 2 * Caf / m
        f = 2 * lf * Caf / Iz

        Ai = np.array([[a, 0, b, 0], [0, 0, 1, 0], [c, 0, d, 0], [1, xdot, 0, 0]])
        Bi = np.array([[e], [0], [f], [0]])
        Ci = np.array([[0, 1, 0, 0], [0, 0, 0, 1]])

        Ad = np.identity(Ai.shape[0]) + Ai*T
        Bd = Bi*T
        Cd = Ci
        return Ad, Bd, Cd

    def final_matrices(self, Ad, Bd, Cd, S, Q, R, Hz):
        A = np.concatenate((Ad, Bd), axis=1)
        temp = np.array([[0, 0, 0, 0, 1]])
        A = np.concatenate((A, temp), axis=0)
        temp = np.array([[1]])
        B = np.concatenate((Bd, temp), axis=0)
        C = np.concatenate((Cd, np.zeros((np.size(Cd, 0), np.size(Bd, 1)))), axis=1)
        CQC = np.matmul(np.transpose(C), Q)
        CQC = np.matmul(CQC, C)
        CSC = np.matmul(np.transpose(C), S)
        CSC = np.matmul(CSC, C)
        QC = np.matmul(Q, C)
        SC = np.matmul(S, C)
        Qbar = np.zeros((1, CQC.shape[1] * Hz))
        Tbar = np.zeros((1, QC.shape[1] * Hz))
        Rbar = np.zeros((1, R.shape[1] * Hz))
        Cbar = np.zeros((1, B.shape[1] * Hz))
        Ahat = np.zeros((1, A.shape[1]))
        for i in range(0, Hz):
            if i == Hz - 1:
                temp1 = np.zeros((CSC.shape[0], (i) * CSC.shape[1]))
                temp2 = np.zeros((CSC.shape[0], (Hz - i - 1) * CSC.shape[1]))
                rows = np.concatenate((temp1, CSC, temp2), axis=1)
                Qbar = np.concatenate((Qbar, rows), axis=0)
                tempt1 = np.zeros((SC.shape[0], (i) * SC.shape[1]))
                tempt2 = np.zeros((SC.shape[0], (Hz - i - 1) * SC.shape[1]))
                rowst = np.concatenate((tempt1, SC, tempt2), axis=1)
                Tbar = np.concatenate((Tbar, rowst), axis=0)
            else:
                temp1 = np.zeros((CQC.shape[0], (i) * CQC.shape[1]))
                temp2 = np.zeros((CQC.shape[0], (Hz - i - 1) * CQC.shape[1]))
                rows = np.concatenate((temp1, CQC, temp2), axis=1)
                Qbar = np.concatenate((Qbar, rows), axis=0)
                tempt1 = np.zeros((QC.shape[0], (i) * QC.shape[1]))
                tempt2 = np.zeros((QC.shape[0], (Hz - i - 1) * QC.shape[1]))
                rowst = np.concatenate((tempt1, QC, tempt2), axis=1)
                Tbar = np.concatenate((Tbar, rowst), axis=0)
            tempr1 = np.zeros((R.shape[0], (i) * R.shape[1]))
            tempr2 = np.zeros((R.shape[0], (Hz - i - 1) * R.shape[1]))
            rowsr = np.concatenate((tempr1, R, tempr2), axis=1)
            Rbar = np.concatenate((Rbar, rowsr), axis=0)
            tempa = np.zeros((B.shape[0], 1))
            tempo = np.zeros((B.shape[0], 1))
            for j in range(Hz):
                if j == i:
                    tempb = B
                elif j < i:
                    tempaa = np.linalg.matrix_power(A, (i-j))
                    tempaa = np.matmul(tempaa, B)
                    tempa = np.concatenate((tempa, tempaa), axis=1)
                else:
                    tempoo = np.zeros((B.shape[0], B.shape[1]))
                    tempo = np.concatenate((tempo, tempoo), axis=1)
            tempa = np.delete(tempa, 0, axis=1)
            tempo = np.delete(tempo, 0, axis=1)
            rowsc = np.concatenate((tempa, tempb, tempo), axis=1)
            Cbar = np.concatenate((Cbar, rowsc), axis=0)
            Aele = np.linalg.matrix_power(A, i + 1)
            Ahat = np.concatenate((Ahat, Aele), axis=0)
        Qbar = np.delete(Qbar, 0, axis=0)
        Tbar = np.delete(Tbar, 0, axis=0)
        Rbar = np.delete(Rbar, 0, axis=0)
        Cbar = np.delete(Cbar, 0, axis=0)
        Ahat = np.delete(Ahat, 0, axis=0)

        Hf = np.matmul(np.transpose(Cbar), Qbar)
        Hf = np.matmul(Hf, Cbar)
        Hbar = Hf + Rbar

        Ff = np.matmul(np.transpose(Ahat), Qbar)
        Ff = np.matmul(Ff, Cbar)
        Fs = np.matmul(-Tbar, Cbar)
        Fbar = np.concatenate((Ff, Fs), axis=0)

        return Hbar, Fbar
