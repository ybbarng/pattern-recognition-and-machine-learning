import numpy as np
import pylab as pl

with np.load('data2_1.npz') as data:
    X1 = data['X1']
    X2 = data['X2']

m1 = np.mean(X1, axis=0)
m2 = np.mean(X2, axis=0)

s1 = np.cov(X1, rowvar=False)
s2 = np.cov(X2, rowvar=False)

print('m1')
print(m1)
print('s1')
print(s1)
print('m2')
print(m2)
print('s2')
print(s2)
