# -*- coding: utf-8 -*-
"""Meta2_1_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uv0wHbJgtZos5d-noZMEeVZb1LPchwYb
"""

import numpy as np
import matplotlib.pyplot as plt

# Carga el dataset
data = np.loadtxt('price_dataset.dat',delimiter=' ')
x = data[:,0:2]
y = data[:,2:4]
# Genera la matriz de diseno
q,_ = np.shape(x)
_,m = np.shape(y)

#Conviene segundo orden puesto que tiene más porcentaje de correlación (R2)
M2 = np.array([x[:,0], (x[:,1]), (x[:,0] * x[:,1]), (x[:,0] * x[:,0]), (x[:,1] * x[:,1])]).T #SEGUNDO ORDEN
#M2 = np.array([x[:,0], (x[:,1]), (x[:,0] * x[:,1]), (x[:,0] * x[:,0]), (x[:,1] * x[:,1]), (x[:,1] * x[:,1] * x[:,0]), (x[:,0] * x[:,0] * x[:,1]), (x[:,0] * x[:,0] * x[:,0]), (x[:,1] * x[:,1] * x[:,1])]).T #TERCER ORDEN

A = np.c_[M2,np.ones((q,1))]
# Optimizacion
ATA = A.T @ A
b = A.T @ y
theta = np.linalg.pinv(ATA) @ b
# Evalua la hipotesis
yh = A @ theta
# Evalua metricas
e = y-yh
ev = e.flatten(order='F')
SSE = ev.T @ ev
MSE = SSE/(q*m)
RMSE = np.math.sqrt(MSE)
# Evalua coeficiente de determinación (𝑅2)
yb = np.mean(y)
e = y-yb
ev = e.flatten(order = 'F')
SST = ev.T @ ev
R2 = 1-SSE/SST

print("SSE   = ",SSE)
print("MSE   = ",MSE)
print("RMSE  = ",RMSE)
print("R2    = ",R2)
print("\nTHETA = ",theta)