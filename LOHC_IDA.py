"""
Created on Sat Jan 18 23:51:40 2025

Author: Saiteja Sistla
PhD Student, University of Canterbury, New Zealand

This script generates hazard-consistent engineering demand parameters (HC-EDPs)
using the Loss-Oriented Hazard-Consistent Incremental Dynamic Analysis (LOHC-IDA)
framework.

The implementation follows the formulation presented in Chapter 4 of the author's
PhD thesis and the associated journal article.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# =============================================================================
# USER INPUTS
# =============================================================================
modelling_unc       = 0.3 # modelling uncertainity (lognormal standard deviation value)
num_realization     = 100 # number for HC-EDPs to be simulated by the LOHC-IDA framework

# =============================================================================
# INPUT DATA
# =============================================================================
"""
Equation (3): Secondary intensity measures (IMs) of scaled ground motions
(e.g., SaRatio and PGA)
"""
IM_GM               = pd.read_excel('data/GM_IMs.xlsx') #IMs of the scaled generic GMs at the intensity level under consideration 

"""
Equation (4): Engineering Demand Parameters (EDPs) obtained from IDA
"""
EDPs                = pd.read_excel('data/EDPs.xlsx') # EDPs of the building at the intensity level under consideration (obtained from IDA)
N_edp               = len(EDPs.iloc[0,:]) # total no of EDPs considered


# =============================================================================
# TARGET INTENSITY MEASURE PARAMETERS FROM PSHA
# =============================================================================
"""
Equation (7a): Conditional mean vector of target secondary IMs (SaRatio and PGA)
"""
Tar_IM_mean  = pd.read_excel('data/Tar_IM_mean.xlsx') # target mean SaR and PGA at the intensity under consideration

"""
Equation (7b): Conditional covariance matrix of target secondary IMs (SaRatio and PGA)
"""
Tar_IM_cov   = pd.read_excel('data/Tar_IM_cov.xlsx', index_col=0) #  2x2 covariance matrix of SaR and PGA


# =============================================================================
# COMPUTATION OF HC-EDP MEAN AND STANDARD DEVIATION
# =============================================================================
HC_EDP_mean = [] # List of hazard-consistent EDP mean values
HC_EDP_std  = [] # List of hazard-consistent EDP standard deviation values
for edpno in range(N_edp): # MVLR for PSDR and PBS
        """
        Equation (5a): Multivariate log-linear regression of EDPs
        conditioned on secondary IMs
        """
        X    = np.log(IM_GM)
        Y    = np.log(EDPs.iloc[:,edpno]) # coverting the EDP values of interest into log space   
        model         = LinearRegression().fit(X, Y) # fitting a multi variable regression model with SaR and PGA conditioned on SaT1 as predictor variables and EDP as the dependent variable
        coefficients  = model.coef_           # calculating the coefficients of regression
        intercept    = model.intercept_

        """
        Equation (8a): Expected value of ln(EDP) at target IM^s
        """
        y_single      = model.predict(np.log(Tar_IM_mean))[0] # expected EDP value calculated based on target SaR and PGA conditioned on SaT1
        
        """
        Equation (5b): Variance of the error term
        """
        Y_pred        = model.predict(X) # Predicted EDP values from the multi variable regression model
        residuals     = Y - Y_pred # calculation of residuals
        Var_1         = np.sum(residuals**2)/(len(X)-X.shape[1]-1) # calculation of the variance in y for a given value of inputs
        
        """
        Equation (8b): Variance contribution due to uncertainty in IM^s
        """
        Var_2         = coefficients.reshape(1, -1) @ Tar_IM_cov @ coefficients.reshape(-1, 1)
        Var_tot       = Var_1 +  Var_2 # total variance in y considering the covariance among inputs     
        HC_EDP_mean.append(y_single)
        HC_EDP_std.append(np.sqrt(Var_tot[0][0]))


# =============================================================================
# INSERT PGA STATISTICS EXPLICITLY
# =============================================================================

"""
PGA is explicitly included to preserve correlation structure during HC-EDP
simulation.
"""
# inserting PGA in the third position after PSDR1x and PSDR 2x
HC_EDP_mean.insert(2, np.log(Tar_IM_mean.iloc[0,1])) # inserting the mean
HC_EDP_std.insert(2, np.sqrt(Tar_IM_cov.iloc[1,1]))  # inserting the standard deviation


# =============================================================================
# EDP CORRELATION MATRIX FROM IDA
# =============================================================================
"""
Equation (6a): Correlation matrix of ln(EDPs) estimated from IDA results
"""
EDPs_new            = EDPs.copy() # creating a copy of the original EDPs
EDPs_new.insert(2, 'PGA', IM_GM.iloc[:,1]) # inserting PGA values for correlation computation
EDP_cor             = np.corrcoef(np.log(EDPs_new).iloc[:,:], rowvar=False) # Calculate the correlation matrix of lnEDPs from IDA


# =============================================================================
# SIMULATION OF HAZARD-CONSISTENT EDPs
# =============================================================================  
np.random.seed(1)       # Set the random seed for reproducibility  of the results

"""
Equation (10a): HC-EDP covariance matrix constructed using
IDA-based correlation and HC-EDP standard deviations
"""
lnEDPs_cov   = np.outer(HC_EDP_std, HC_EDP_std)*EDP_cor # Calculate the HC-covariance matrix of lnEDPs using EDP correlation from IDA and HC-std. dev
Beta1        = np.array([modelling_unc, 0.0])
lnEDPs_cov_rank = np.linalg.matrix_rank(lnEDPs_cov) # computing the rank of the covariance matrix. 

'If the covariance matrix is not full rank, it has to be reconstructured using'
'The non-zero eigen values and their respective eigen vectors to allow the simulation of HC-EDPs' 

"""
Variance inflation to account for modelling uncertainty
"""
sigma = np.sqrt(np.diag(lnEDPs_cov))
sigmap2 = sigma**2
# Inflate variance to account for moedlling uncertainity
Beta1 = Beta1.reshape(1, -1)
sigmap2 += Beta1[:, 0]**2
sigmap2 += Beta1[:, 1]**2
sigma = np.sqrt(sigmap2)
sigma2 = np.outer(sigma, sigma)
# Compute inflated covariance matrix
lnEDPs_cov_inflated = EDP_cor * sigma2

# =============================================================================
# HANDLING RANK-DEFICIENT COVARIANCE MATRICES
# =============================================================================
"""
Eigen-decomposition is used to reconstruct the covariance matrix
when it is not full rank.
"""
# Compute eigenvalues and eigenvectors of the covariance matrix
Uvecs, Svals, Vt     = np.linalg.svd(lnEDPs_cov_inflated)
eigenvalues  = Svals # eigen values
eigenvectors = Uvecs # eigen vectors
# Sort eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)
eigenvalues    = eigenvalues[sorted_indices]
eigenvectors   = eigenvectors[:, sorted_indices]
# Partition eigenvectors and eigenvalues to remove non-zero eigen values and their respective eigen vectors
if lnEDPs_cov_rank >= len(eigenvalues):
    L_use = eigenvectors
    D_use = np.diag(np.sqrt(eigenvalues))
else:
    L_use = eigenvectors[:, -lnEDPs_cov_rank:]
    D_use = np.diag(np.sqrt(eigenvalues[-lnEDPs_cov_rank:]))
# Generate standard normal random numbers
if lnEDPs_cov_rank >= len(eigenvalues):
    U = np.random.randn(num_realization, len(eigenvalues))
else:
    U = np.random.randn(num_realization, lnEDPs_cov_rank)
# Create Lambda matrix
Lambda = np.dot(L_use, D_use)


# =============================================================================
# FINAL HC-EDP SIMULATION
# =============================================================================
simulated_demands_log = np.dot(Lambda, U.T) + np.array(HC_EDP_mean).reshape(-1, 1) # simulated demands in log space
simulated_demands     = np.exp(simulated_demands_log) # simulated demands in linear space




