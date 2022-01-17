"""
an implementation of quasi-discrete Hankel transform. 

Reference: Li Yu, et.al, Quais-discrete Hankel transform, Optics Letters, 23, 409, 1998. 

@author: Bingbing Zhu 
@date: 2021/12/19 
""" 

import numpy as np 
from scipy.special import jn_zeros, j0, j1, jv  


class Hankel_qDHT:
    """
    an implementation of quasi-discrete Hankel transform. 
    
    R1 = R2 == R is assumed. 

    Reference: Li Yu, et.al, Quais-discrete Hankel transform, Optics Letters, 23, 409, 1998. 
    """
    
    def __init__(self, N=1000):
        """
        N: total nodes of used Bessel 0-order function. 
        
        generate: 
        .r: 1D-array, the corresponding radius axis data. 
        ._S: _S = j_(N), the (N+1)-th zero point of J_0(x) function. 
        .js: the N+1 positive zeros of J_0(x) function. 
        .j1_inv: the function value of 1/(J_1(x)) for x at each .js. 
        .R: = (._S/2/pi)**0.5, the 
          .r = [j_1, j_2, j_3, ..., j_N ] /(2pi*R) 
        .C: 2D matrix, symmetric, C_ij = 2/S*J_0(j_i*j_j/S) / |J_1(j_i)| / |J_1(J_j)|
        .F_multiplier" 1D array, F_multiplier_i = R/|J_1(j_i)| 
        """
        self.N = N 
        self.js = jn_zeros(0, self.N + 1) 
        
        self._S = self.js[-1] # j_(N+1) 
        self.R = (self._S/2.0/np.pi)**0.5 # r's cutoff position. 
        self.r = self.js[0:-1] / (2.0*np.pi*self.R) # the r axis for the field. 
        
        self.j1_inv = 1.0 / np.abs(j1(self.js)) 
        self.F_multiplier = self.j1_inv[0:-1]*self.R 
        self.F_multiplier_inv = 1.0 / self.F_multiplier
        
        # 1 / |J_1(n)*J_1(m)|: 
        J1_inv_mesh_x, J1_inv_mesh_y = np.meshgrid(self.j1_inv[0:-1]*(2.0/self._S), self.j1_inv[0:-1])  
        #self.Cmn =  J1_inv_mesh_x * J1_inv_mesh_y * j0(np.outer(self.js[0:-1], self.js[0:-1]/self._S) ) # 
        self.Cmn =  np.outer(self.j1_inv[0:-1]*(2.0/self._S), self.j1_inv[0:-1]) * j0(np.outer(self.js[0:-1], self.js[0:-1]/self._S) ) # 


    def transform(self, f1):
        r"""
        perform 0-order Hankel transform. f2(r2) = 2pi*\int_{0}^{+\infty} f1(r1)*J_0(2pi*r1*r2)*r1*dr1 
        
        return f2 
        
        Note: f1 must defined in self.r axis. 
        len(f1)==len(self.r)==self.N, and each r-point corresponding to J_0(x)'s zeros jn by: 
        rn = jn/(2*pi*R), wher 2*pi*R^2==j_(N+1)~(N+1)*pi 
        ==> R ~ \sqrt((N+1)/2)  
        
        f1: 1D-array, float or complex valued. length == self.N, defined on self.r axis. 
        """ 
        if len(f1) != self.N: 
            print("invalid f1") 
            return None 
        f2 = self.F_multiplier_inv * np.matmul(self.Cmn, self.F_multiplier*f1)  # transform, 
        
        return f2 
    

# test and compare: 
if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    import time 
    def timming(func, total=100, name=""):
        start = time.time()
        for _ in range(total):
            func() 
        end = time.time() 
        print(name, "total time [sec]: ", end-start) 
    #

    N = 4096 
    # qDHT: 
    ht = Hankel_qDHT(N)  
    r = ht.r 
    
    # sum as integral: 
    rho = np.linspace(0., ht.R, N)  # start from 0.0 or ht.r[0], doesn't matter much. 
    drho = (ht.R - 0.0) / (N - 1.0) 
    c1 = 2.0*np.pi*drho * rho  # 2pi*r*dr 
    c2 = j0(np.outer(2.*np.pi*rho, rho) ) # J_0(r*r') 
    
    
    # field: 
    a, c = 1.0, 0.2 
    field = lambda r: np.exp(-np.pi*r**2/(a+1.0J*c))  
    ft_theory = lambda r:  (a+1.0J*c)*np.exp(-np.pi*(a+1.0J*c)*r**2)   
    # 1.0 
    f_qDHT = ht.transform(field(ht.r)) 
    # 2.0 
    f_sum = np.matmul(c2, c1*field(rho)) 
    
    # display: 
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    ax.plot(ht.r, np.abs(f_qDHT), "b-", label="qDHT") 
    ax.plot(rho, np.abs(f_sum), "g-", label="sum") 
    ax.plot(rho, np.abs(ft_theory(rho)), "r-", label="theory") 
    ax.set_yscale("log") 
    ax.set_ylim((1E-30, 1E3))
    plt.title("Hankel_0{$e^{\\frac{-\pi r^2}{%.2f+%.2fJ}}$ }"%(a, c))
    plt.legend() 
    plt.show() 
    
    
    print(np.amax(np.matmul(ht.Cmn, ht.Cmn)-np.eye(ht.N))) 
    # Timming: 
    t1 = lambda : ht.transform(field(ht.r))  
    t2 = lambda :  np.matmul(c2, c1*field(rho))  
    timming(t1, 20, name="qDHT") 
    timming(t2, 20, name="sum") 

