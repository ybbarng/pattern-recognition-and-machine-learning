import numpy as np
import pylab as pl
from scipy.linalg import sqrtm


N = 25  # Number of data
m1 = [3, 5]
m2 = [5, 3]
i1 = np.tile(m1, (N, 1))  # Initial matrix of X1 with mean m1
i2 = np.tile(m2, (N, 1))  # Initial matrix of X2 with mean m2

s1 = np.mat('1 1; 1 2')  # Covariance matrix of X1
s2 = np.mat('1 1; 1 2')  # Covariance matrix of X2

print('m1')
print(m1)
print('s1')
print(s1)
print('m2')
print(m2)
print('s2')
print(s2)

# Create randomized dataset with each mean and each covariance
X1 = np.random.randn(N, 2).dot(sqrtm(s1)) + i1
X2 = np.random.randn(N, 2).dot(sqrtm(s2)) + i2

# Plot the dataset
pl.plot(X1[:,0],X1[:,1], '+')
pl.plot(X2[:,0],X2[:,1], 'd')
pl.grid('on')
pl.show()

# Save the dataset
np.savez('data2_1', X1=X1, X2=X2)
