#!/usr/bin/env python3
# -*- coding: utf-8 -*-

@author: ariel
"""
import numpy as np
import matplotlib.pyplot as plt


from GaussianFunc import affineGaussianTransform, approxGaussianTransform 

from GaussianFunc import SigmaEllipse3, principal_components

############################## TASK 1 a ######################################

# Given Data : ----------------------------------------------------
# ffine Gaussian Transform
A = np.array([[1, 1], [1, -1]])
mu_x = np.array([10, 0])
Sigma_x = np.array([[0.2, 0], [0, 8]])
b = None
n = 100
level = 3

mu_yaf, Sigma_yaf = affineGaussianTransform(mu_x, Sigma_x, A, b)



# Approximate Gaussian Transform
sample_num = 500

# Transformation function for non-linear 
def h(x):
    h = A @ x
    return h

mu_yap, Sigma_yap, sample_y = approxGaussianTransform(mu_x, Sigma_x, h, sample_num)


# Plotting : ----------------------------------------------------


# Plot Ellipses
ellipse_af = SigmaEllipse3(mu_yaf, Sigma_yaf, level=3, n=100)  # Affine transform ellipse
ellipse_ap = SigmaEllipse3(mu_yap, Sigma_yap, level=3, n=100)  # Approximate transform ellipse



fig, ax = plt.subplots()

ax.plot(ellipse_af[0], ellipse_af[1], 'b--', label='3-Sigma Ellipse (Affine)')


ax.plot(ellipse_ap[0], ellipse_ap[1], label='3-Sigma Ellipse (Approximate)', color='green')

ax.scatter(sample_y[:, 0], sample_y[:, 1], s=20, color='blue', alpha=0.7, label='Approximate Samples')

ax.scatter(mu_yap[0], mu_yap[1], color='red', label='Mean')

ax.scatter(mu_yaf[0], mu_yaf[1], color='black', label='Affline Mean')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Gaussian Transform Methods')
ax.set_aspect('equal')
ax.grid(True)
ax.legend()

plt.show()

print('------------------------------------')
print('Affine Gaussian Transform')
print('mu_y :', mu_yaf)
print('S_y :', Sigma_yaf)

print('------------------------------------')
print('Approximate Gaussian Transform')

print('mu_y :', mu_yap)
print('S_y :', Sigma_yap)

# Plot 3-sigmaellips from defined functions


#%%
############################## TASK 1b ######################################

# Radar sensor : angle phi (radian), distance (m), 

A = np.array([[1, 1], [1, -1]])
mu_x = np.array([10, 0])
Sigma_x = np.array([[0.2, 0], [0, 8]])
b = None


# Transformation function for non-linear 
def h(x):
    rho = np.sqrt(x[0]**2 + x[1]**2)  
    phi = np.arctan2(x[1], x[0])  
    return np.array([rho, phi])



fig, axes = plt.subplots(2, 2, figsize=(12, 12))


sample_nums = [15, 100, 1000, 10000]
for i, sample_num in enumerate(sample_nums):
    
     
    mu_y2, Sigma_y2, sample_y2 = approxGaussianTransform(mu_x, Sigma_x, h, sample_num)
    ellipse= SigmaEllipse3(mu_y2, Sigma_y2, level=3, n=100)
    
    # Plot on each subplot
    row, col = divmod(i, 2) 
    ax = axes[row, col]
    ax.plot(ellipse[0], ellipse[1], label=f'3-Sigma Ellipse (sample_num={sample_num})')
    ax.scatter(sample_y2[:, 0], sample_y2[:, 1], s=20, color='blue', alpha=0.7, label='Samples')
    ax.scatter(mu_y2[0], mu_y2[1], color='black', label='Mean')

    ax.set_title(f'sample_num={sample_num}')
    ax.set_xlabel('rho (distance)')
    ax.set_ylabel('phi (angle in radians)')
    
    ax.set_xlim(8, 14)  
    ax.set_ylim(-1, 1)  
    
    ax.set_aspect('equal')
    ax.grid(True)

# layout: better spacing between plots
fig.tight_layout()
plt.show()


print('------------------------------------')
print('Approximate Gaussian Transform in Polar Coordinates')
print('mu_y (rho, phi):', mu_yap)
print('Sigma_y (Covariance):', Sigma_yap)


#%%
############################## TASK 2a ######################################

# mu_Ha = 1.1m snow ,mu_Kv = 1.0 m snow 
# Anna = y =x+r , r = mu = 0 ,sigma = 0.2^2
# Else = mu=0 , Sigma = 1^2
from GaussianFunc import JointGaussian, SigmaEllipse3, principal_components
import matplotlib.pyplot as plt

sigma_x = 0.5**2 
A_xr = np.array([[1,1],[1,0]])
b_xr = np.array([0,0])


mu_x_Anna = 1.1
mu_r_Anna = 0
y_Anna = 1.0
sigma_r_Anna = 0.2**2

mu_x_Elsa = 1.0
mu_r_Elsa = 0
y_Elsa = 2.0        # observation of Elsa
sigma_r_Elsa = 1**2 

# ------------- Anna

mu_Anna, sigma_Anna = JointGaussian(mu_x_Anna, mu_r_Anna, sigma_x, sigma_r_Anna, A_xr, b_xr)
ellipse_Anna = SigmaEllipse3(mu_Anna, sigma_Anna, level =3, n =100)
#computing the eigen values and vectors based on joint gaussian
eigval_Anna, eigvec_Anna = principal_components(mu_Anna, sigma_Anna)

# ------------- Elsa
mu_Elsa, sigma_Elsa = JointGaussian(mu_x_Elsa, mu_r_Elsa, sigma_x, sigma_r_Elsa, A_xr, b_xr)
ellipse_Elsa = SigmaEllipse3(mu_Elsa, sigma_Elsa, level=3, n=100)
eigval_Elsa, eigvec_Elsa = principal_components(mu_Elsa, sigma_Elsa)


#fig, ax = plt.subplots()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


# Plot For Anna =================================================================

ax1.plot(ellipse_Anna[0], ellipse_Anna[1], label='Ellipse_Anna')
ax1.scatter(mu_Anna[0], mu_Anna[1], color='black', label='Mu_Anna')

for i in range(2):  # 2D data, so two eigenvectors
    ax1.quiver(mu_Anna[0], mu_Anna[1], eigvec_Anna[0, i] * np.sqrt(eigval_Anna[i]),
              eigvec_Anna[1, i] * np.sqrt(eigval_Anna[i]), angles='xy', scale_units='xy', scale=1, color='blue', width=0.005)
    
ax1.grid(True)
ax1.legend()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Anna\'s 2D Gaussian Ellipse')

# For Elsa =================================================================
ax2.plot(ellipse_Elsa[0],ellipse_Elsa[1], label='Ellipse_Elsa')
ax2.scatter(mu_Elsa[0], mu_Elsa[1], color='red', label='Mu_Elsa')

for i in range(2):
    ax2.quiver(mu_Elsa[0], mu_Elsa[1], eigvec_Elsa[0, i] * np.sqrt(eigval_Elsa[i]),
              eigvec_Elsa[1, i] * np.sqrt(eigval_Elsa[i]), angles='xy', scale_units='xy', scale=1, color='green', width=0.005)

ax2.grid(True)
ax2.legend()
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Elsa\'s 2D Gaussian Ellipse')

plt.tight_layout()
plt.show()
#%%
############################## TASK 2b ######################################

from GaussianFunc import posteriorGaussian


x_values = np.linspace(-1, 4, 100)


# Computing the posteriori and PDF for Anna
mu_pos_Anna, sigma_pos_Anna = posteriorGaussian(mu_x_Anna, sigma_x, sigma_r_Anna, y_Anna)
PDF_Anna = 1/np.sqrt(2*np.pi*sigma_pos_Anna) * np.exp((-0.5*(x_values-mu_pos_Anna)*(x_values-mu_pos_Anna))/sigma_pos_Anna)
#pdf_Anna = norm.pdf(x_values, mu_pos_Anna, sigma_pos_Anna)   #optional : from scipy.stats import norm
                                                              # alt. for this PDF/CDF equation import norm 



# Computing the posteriori and PDF for Elsa

mu_pos_Elsa, sigma_pos_Elsa = posteriorGaussian(mu_x_Elsa, sigma_x, sigma_r_Elsa, y_Elsa)
PDF_Elsa = 1/np.sqrt(2*np.pi*sigma_pos_Elsa) * np.exp((-0.5*(x_values-mu_pos_Elsa)*(x_values-mu_pos_Elsa))/sigma_pos_Elsa)
#pdf_Elsa = norm.pdf(x_values, mu_pos_Elsa, sigma_pos_Elsa)



# plotting 1D density using Posetiori 


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 5))


ax1.plot(x_values, PDF_Anna, color='blue', label='Anna Posterior')
ax1.scatter(mu_pos_Anna, 0, color='black', label='Mu_pos_Anna')
ax1.set_xlabel('Snow Depth (m)')
ax1.set_ylabel('Probability Density')

ax1.legend()


ax2.plot(x_values, PDF_Elsa, color='blue', label='Elsa Posterior')
ax2.scatter(mu_pos_Elsa, 0, color='red', label='Mu_pos_Elsa')




ax2.set_xlabel('Snow Depth (m)')
ax2.set_ylabel('Probability Density')
ax2.legend()

ax.grid(True)
plt.show()

#%%
############################## TASK 2c ######################################
# Maximum Expectation to find the most snow 

# Desicion Making based on Expectation
if mu_pos_Anna > mu_pos_Elsa:
    print("Anders should go to Hafjell (based on higher expected snow depth).")
else:
    print("Anders should go to Kvitfjell (based on higher expected snow depth).")
    
# Desicion Making based on Standard Deviation / spread

# Print standard deviations (uncertainty)
print(f"Standard deviation (spread) for Hafjell (Anna): {sigma_pos_Anna}")
print(f"Standard deviation (spread) for Kvitfjell (Elsa): {sigma_pos_Elsa}")

# 3-sigma range (99.7% confidence interval)
print(f"3-Sigma range for Hafjell: [{mu_pos_Anna - 3*sigma_pos_Anna}, {mu_pos_Anna + 3*sigma_pos_Anna}]")
print(f"3-Sigma range for Kvitfjell: [{mu_pos_Elsa - 3*sigma_pos_Elsa}, {mu_pos_Elsa + 3*sigma_pos_Elsa}]")


#%%
############################## TASK 3 ######################################
from scipy.stats import norm
from GaussianFunc import mmse


theta = np.linspace(-10, 10, 200)

#p = norm.pdf(theta, mu , np.sqrt(sigma))  : Posterior density 
p_a = 0.1 * norm.pdf(theta, 1, np.sqrt(0.5)) + 0.9 * norm.pdf(theta, 1, np.sqrt(9))
p_b = 0.49 * norm.pdf(theta, 5, np.sqrt(2)) + 0.51 * norm.pdf(theta, -5, np.sqrt(2))
p_c = 0.4 * norm.pdf(theta, 1, np.sqrt(2)) + 0.6 * norm.pdf(theta, 2, np.sqrt(1))

MAP_a = np.max(p_a)

map_a = theta[np.argmax(p_a)]    #argmax : Returns the indices of the maximum values along an axis.
map_b = theta[np.argmax(p_b)]
map_c = theta[np.argmax(p_c)]


mmse_a = mmse(np.array([0.1, 0.9]), np.array([1,1]))
mmse_b = mmse(np.array([0.49, 0.51]), np.array([5,-5]))
mmse_c = mmse(np.array([0.4, 0.6]), np.array([1,2]))

y_a = np.interp(mmse_a, theta, p_a)
y_b = np.interp(mmse_b, theta, p_b)
y_c = np.interp(mmse_c, theta, p_c)

fig, axes = plt.subplots(1,3, figsize=(15,10))

axes[0].plot(theta, p_a)
axes[0].scatter(map_a, p_a[np.argmax(p_a)], label=r'$MAP_a$')
axes[0].scatter(mmse_a, y_a, label=r'$MMSE_a$')
axes[0].vlines(map_a, ymin=0, ymax=np.max(p_a), color='black', linestyle='--', label='MAP')
axes[0].vlines(mmse_a, ymin=0, ymax=y_a, color='red', linestyle='--', label='MMSE')
axes[0].set_title("Posterior p_a")
axes[0].grid(True)


axes[1].plot(theta, p_b)
axes[1].scatter(map_b, p_b[np.argmax(p_b)],label=r'$MAP_b$')
axes[1].scatter(mmse_b, y_b,label=r'$MMSE_b$')
axes[1].vlines(map_b, ymin=0, ymax=np.max(p_b), color='black', linestyle='--', label='MAP')
axes[1].vlines(mmse_b, ymin=0, ymax=y_b, color='red', linestyle='-', label='MMSE')
axes[1].set_title("Posterior p_b")
axes[1].grid(True)


axes[2].plot(theta, p_c)
axes[2].scatter(map_c, p_c[np.argmax(p_c)],label=r'$MAP_c$')
axes[2].scatter(mmse_c, y_c,label=r'$MMSE_c$')
axes[2].vlines(map_c, ymin=0, ymax=np.max(p_c), color='blue', linestyle='--', label='MAP')
axes[2].vlines(mmse_c, ymin=0, ymax=y_c, color='red', linestyle='-', label='MMSE')
axes[2].set_title("Posterior p_c")
axes[2].grid(True)

for ax in axes:
    ax.legend()

plt.tight_layout()
plt.show()

























