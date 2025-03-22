import numpy as np
from scipy.stats import norm

def Black76Call(F, K, sigma, T, D=1.0):
    """
    Black Model

    Args:
    F: forward price
    K: strike price
    sigma: implied vol
    T: tenor (years)
    D: numeraire. e.g DF, PV01, etc.

    Returns:
    call option price
    """
    d1 = (np.log(F/K)+0.5*sigma*sigma*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return D*(F*norm.cdf(d1)-K*norm.cdf(d2))

def Black76Put(F, K, sigma, T, D=1.0):
    return Black76Call(F, K, sigma, T, D) - D*(F-K)

def DisplacedDiffusionCall(F, K, sigma, beta, T, D):
    return Black76Call(F/beta, K+F*(1-beta)/beta, sigma*beta, T, D)

def DisplacedDiffusionPut(F, K, sigma, beta, T, D):
    return Black76Put(F/beta, K+F*(1-beta)/beta, sigma*beta, T, D)

def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom
    return sabrsigma

