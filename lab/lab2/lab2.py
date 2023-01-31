import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
import math


class UOM():
    def __init__(self, epochs, Lambda):
        self.epochs = epochs
        self.Lambda = Lambda


    def obj(self, x):
        return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2


    def grad(self,x):
        g = np.zeros((2, 1))
        g[0, 0] = 400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1)
        g[1, 0] = -200 * (x[0] ** 2 - x[1])
        return g


    def h(self, x):
        return np.matrix([[1200 * x[0] * x[0] - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])

    def Wolfe_Powell(self, x, alpha_, f, g, d, rho, sigma):
        epochs = 800
        a1 = 0
        a2 = alpha_
        alpha = (a1 + a2) / 2
        phi0, phi0_ = f, np.dot(g.T[0], d)[0]
        phi1, phi1_ = phi0, phi0_
        
        for i in range(epochs):
            phi = self.obj(x + alpha * d.T[0])
            
            if phi <= phi0 + rho * alpha * phi0_:
                g_new = self.grad(x + alpha * d.T[0])
                phi_ = np.dot(g_new.T[0], d)[0]
                
                if phi_ >= sigma * phi0_:
                    break
                elif abs(phi1_ - phi_) < 1e-32:
                    break
                else:
                    alpha_new = alpha + (alpha - a1) * phi_ / (phi1_ - phi_)
                    a1 = alpha
                    alpha = alpha_new
                    phi1 = phi
                    phi1_ = phi_
            else:
                alpha_new = a1 + 0.5 * (a1 - alpha) ** 2 * phi1_ / ((phi1 - phi) - (a1 - alpha) * phi1_)
                alpha = alpha_new
        return alpha

    

    def Quasi_Newton(self, x0, func):
        l = np.zeros(self.epochs + 1)
        x = np.zeros((self.epochs + 1, len(x0)))
        x[0] = x0
        l[0] = self.obj(x[0])
        H_k = np.eye(len(x0))
        epochs = 0
        for i in range(self.epochs):
            g_k = self.grad(x[i])
            d_k = -1.0 * np.dot(H_k, g_k)
            alpha = self.Wolfe_Powell(x=x[i], alpha_=1, f=l[i], g=g_k, d=d_k, rho=0.25, sigma=0.5)
            x[i + 1] = x[i] + alpha * d_k.T[0]

            l[i + 1] = self.obj(x[i + 1])
            if np.abs(l[i + 1] - l[i]) <= self.Lambda:
                break

            sk = x[i + 1] - x[i]
            sk = sk.reshape((2, 1))
            yk = self.grad(x[i + 1]) - g_k
            if func == 'DFP':
                if np.dot(sk.T, yk) > 0:
                    Hyy = np.dot(np.dot(H_k, yk), yk.T)
                    yHy = np.dot(np.dot(yk.T, H_k), yk)
                    sy = np.dot(sk.T, yk)
                    H_k = H_k - np.dot(Hyy, H_k) / yHy + np.dot(sk, sk.T) / sy
            elif func == 'BFGS':
                syt = np.dot(sk, yk.T)
                yHy = np.dot(np.dot(yk.T, H_k), yk)
                Hy = np.dot(H_k, yk)
                sy = np.dot(sk.T, yk)
                H_k = H_k + (1 + yHy / sy) * np.dot(sk, sk.T) / sy - (np.dot(Hy, sk.T) + np.dot(syt, H_k)) / sy
            epochs = i + 1
        return epochs, x, l




if __name__ == "__main__":
    method = input("1.DFP法\n"
                   "2.BFGS法\n")
    x0 = input("初始点：").split()
    
    x0[0], x0[1] = float(x0[0]), float(x0[1])
    uom = UOM(epochs=300, Lambda=5e-15)
    if method == '1':
        epochs, x, L = uom.Quasi_Newton(x0, func='DFP')
    elif method == '2':
        epochs, x, L = uom.Quasi_Newton(x0, func='BFGS')

    print("epoch = {}".format(epochs))
    print("x1={},x2={},f={}".format(x[epochs][0], x[epochs][1],L[epochs]))
    