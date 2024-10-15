" Transformation of Gaussian random variables"

"position in Cartesian xy-coordinates = x -> not able to be measured directly : y"

import numpy as np
import matplotlib.pyplot as plt

############################## TASK 1 a ######################################
def affineGaussianTransform(mu_x, Sigma_x, A,b=None):
    # y = Ax + b 
    # N(mu_x, S_x)
    # mu_x = mean of x   = A*mu_x+b                 [nx1]  
    # S_x = Covariance of x  = A*S_x*A^T            [nxn]
    # A = linear transform matrix A                 [mxn]
    # b = constant part of transform                [mx1]
    
    # Creates a zero vector with length equal to the number of rows of A
    if b is None:
        b = np.zeros(A.shape[0])  
    
    # Mean 
    mu_y = A @ mu_x + b
    # Covariance 
    Sigma_y = A @ Sigma_x @ A.T

    
    #output : mu_y , S_y 
    return mu_y, Sigma_y


def approxGaussianTransform(mu_x, Sigma_x, h, sample_num):
    
    # approximate the Gaussian transform using sampling for ...
    # ... a non-linear transformation.
    
    # h = function that applies a transformatio to each sample
    # sample_num: number of samples
    
    if sample_num < 10:
        sample_num = 600
        
    
    # a random sample from a multivariate normal distribution
    # mean [nx1], cov [nxn]
    sample_x = np.random.multivariate_normal(mu_x, Sigma_x, sample_num)
    
    # applying h(x) to all samples
    sample_y = np.array([h(x) for x in sample_x])
    
    #approximate mu_y and S_y
    # axis = 0 gives the mean of each column
    # rowvar = False : it says that each row = a sample and each column = variables/feature
        # so np.cov is going to calculate the cov btw features across all samples. 
    mu_y = np.mean(sample_y, axis = 0)
    Sigma_y = np.cov(sample_y, rowvar = False)
    
    
    return mu_y, Sigma_y, sample_y
                     


# This function visulize the confidence ellipse            

def SigmaEllipse3(mu, Sigma, level, n):
    
    # mu = Mean [nx1]
    # S = Covariance matrix [nxn]
    # ax = where to draw the ellipse, matplot axis object
    # c = level = number of standard deviation, here = 3
    # n = number of points to generate
    
    if n < 10:
        n = 50
        
   # We could use the eigen velue /vector but here we use the simple method
    

    # Ellipse function
    # this cholesky transform the circle to make it as ellipse with L-transform matrix
    sqrt_Sigma = np.linalg.cholesky(Sigma)
    
    theta = np.linspace(0, 2*np.pi, n)
    unit_circle = np.array([np.cos(theta),np.sin(theta)])
    
    mu = mu.reshape(-1, 1)
                            
    ellipse_points =mu + level * sqrt_Sigma @ unit_circle
   
    
    # Add the ellipse to the plot. 
    #ax.add_patch(ellipse)
    return  ellipse_points


############################## TASK 2  ######################################
def JointGaussian(mu_x, mu_r, sigma_x, sigma_r, A_xr, b_xr):
    
    # x = true snow depth
    # mu = forecast 
    # r = noise
    # mu_j, sigma_j = measured mu and sigma using joint gaussian
    
    mu_xr = np.array([mu_x, mu_r])
    sigma_xr = np.array([[sigma_x, 0], [0, sigma_r]])
    
    mu_j , sigma_j = affineGaussianTransform(mu_xr, sigma_xr, A_xr, b_xr)
    
    
    return mu_j, sigma_j

def posteriorGaussian(mu_x, sigma_x, sigma_r, y):
    
    # mu_x = prior : forecast
    # sigma_x = forecast 
    # sigma_r = measeured sigma from Anna and Elsa
    # y = observation of snow depth by Anna and Elsa 
    # mu_po, sigma_pos = new mu and sigma
    
    
    mu_pos = ((sigma_r**2 * mu_x) + (sigma_x**2 * y))/ (sigma_x**2 + sigma_r**2)
    sigma_pos = (sigma_x**2 * sigma_r**2)/ (sigma_x**2 + sigma_r**2)
    
    return mu_pos, sigma_pos 


def principal_components(mu, sigma, level=3, ax=None):
    """
    Function to calculate and plot the principal components (eigenvectors).

    Parameters:
    mu: Mean of the Gaussian (center of the ellipse).
    sigma: covariance matrix (2x2).
    level: Number of standard deviations (e.g., 3 for a 3-sigma ellipse).
    ax: Optional, matplotlib axis to plot on.

    Returns:
    eigenvalues: Sorted eigenvalues.
    eigenvectors: Corresponding sorted eigenvectors.
    """
    
    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    
    # Sort eigenvalues and corresponding eigenvectors in descending order
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    return eigenvalues, eigenvectors  # Make sure the function returns these values

############################## TASK 3  ######################################
def mmse(w, mui):
    # w = weights
    #mui = means
    x_hat = np.sum(w * mui)
    return x_hat