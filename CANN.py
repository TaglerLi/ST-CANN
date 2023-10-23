import numpy as np
import matplotlib.pyplot as plt

class CANN:
    def __init__(self,scale):
        self.scale = scale
        self.neuronnum = 200
        self.tau = 50e-5
        self.dt = 200e-6
        self.k = 0.28
        self.a = 3.14 / 10
        self.neurons = np.linspace(-3.14, 3.14, self.neuronnum + 1)
        self.neurons = self.neurons[0:-1]
        self.J = np.zeros((self.neuronnum, self.neuronnum))
        for i in range(self.neuronnum):
            for j in range(self.neuronnum):
                self.J[i][j] = self.gauss(self.neurons[i], self.neurons[j], 1, self.a)

        self.U = self.gauss(self.neurons, 0, 0, 2 * self.a)
        self.p = np.ones( self.neuronnum) * 0.8
        self.R = self.R_com()

    def gauss(self, x, x0, alpha, a):
        return alpha * np.exp(-(x - x0) ** 2 / (2 * a ** 2))

    def R_com(self):
        self.U[self.U < 0] = 0
        R = self.U ** 2 / (1 + self.k * np.dot(self.U, self.U))
        return R

    def mapping(self, xin, flag, vmin, vmax):
        a = 0.99942
        xc = 8.02749
        k = -1.21674
        x1 = a/(1+np.exp(-k*(vmin-xc)))
        x2 = a/(1+np.exp(-k*(vmax-xc)))
        y1 = 1
        y2 = 0
        if flag == 1:
            yout = (y2-y1)*(xin-x1)/(x2-x1)+y1
        else:
            yout = (xin-y1)/(y2-y1)*(x2-x1)+x1
        return yout

    def get_parameter(self, V, initial):
        tau_std = np.zeros(self.neuronnum)
        y0 = np.zeros(self.neuronnum)
        for i in range(len(initial)):
            if initial[i] == 1:
                tau_std[i] = 0.0011
                y0[i] = 0.50343+0.49667*initial
            else:
                tau_std[i] = 0.005926*np.exp(-0.8083*V[i])+0.004379
                a = 0.99942
                xc = 8.02749
                k = -1.21674
                y0[i] = a/(1 + np.exp(-k * (V[i] - xc)))
        return tau_std, y0

    def predict(self, x):
        x = x / (self.scale / 2 / 3.14) - 3.14
        if x == -3.14:
            sti = self.gauss(self.neurons, x, 0, 2 * self.a)
        else:
            sti = self.gauss(self.neurons, x, 0.5, 2 * self.a)
        dU = self.dt / self.tau * (-self.U + np.dot(self.R * self.p, self.J) + sti)
        self.U = self.U + dU

        V = self.R*25+1
        initial = self.mapping(xin = self.p, flag = 2, vmin = 1, vmax = 3.5)
        [tau_std, y0] = self.get_parameter(V = V, initial = initial)
        dinitial = 0.01*self.dt/tau_std*(y0-initial)
        initial = initial+dinitial
        self.p = self.mapping(xin = initial, flag = 1, vmin = 1, vmax = 3.5)

        self.R = self.R_com()
        maxpos = np.where(self.U == np.max(self.U))
        # print(maxpos)
        pre = (maxpos[0]) * (self.scale / self.neuronnum)
        # prex = int(maxpos[0]) + 1

        return pre
