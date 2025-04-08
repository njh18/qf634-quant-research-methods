# %%
import pandas as pd
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm

# %%
# Data required from previous parts

# 1. Discount Curve (OIS & LIBOR)
with open('data/ois_discount_curve.pkl', 'rb') as f:
    ois_discount_curve = pickle.load(f)
with open('data/libor_discount_curve.pkl', 'rb') as f:
    libor_discount_curve = pickle.load(f)

# 2. Calibrated SABR params
__tenors__ = [1,2,3,5,10]
__expiries__ = [1,5,10]
sabr_alpha_df = pd.read_excel("data/sabr_calibrated.xlsx", sheet_name="sabr_alpha", index_col=0)
sabr_alpha_df.index = __expiries__
sabr_alpha_df.columns = __tenors__
sabr_rho_df = pd.read_excel("data/sabr_calibrated.xlsx", sheet_name="sabr_rho", index_col=0)
sabr_rho_df.index = __expiries__
sabr_rho_df.columns = __tenors__
sabr_nu_df = pd.read_excel("data/sabr_calibrated.xlsx", sheet_name="sabr_nu", index_col=0)
sabr_nu_df.index = __expiries__
sabr_nu_df.columns = __tenors__

# 3. CMS rates
cms_rates_df = pd.read_csv("data/cms_rate.csv", index_col=0)
cms_rates_df.columns = cms_rates_df.columns.astype('int64')


# %%
# Option pricers

def Black76Call(F, K, sigma, T, D=1.0):
    d1 = (np.log(F/K)+0.5*sigma*sigma*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return D*(F*norm.cdf(d1)-K*norm.cdf(d2))

def Black76Put(F, K, sigma, T, D=1.0):
    return Black76Call(F, K, sigma, T, D) - D*(F-K)

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

# %%
def IRR_0(K, m, N):
    return 1/K * ( 1.0 - 1/(1 + K/m)**(N*m) )

def IRR_1(K, m, N):
    return -1/K*IRR_0(K, m, N) + 1/(K*m)*N*m/(1+K/m)**(N*m+1)

def IRR_2(K, m, N):
    return -2/K*IRR_1(K, m, N) - 1/(K*m*m)*(N*m)*(N*m+1)/(1+K/m)**(N*m+2)

def g_0(K, p, q):
    return K**(1.0/p)-0.04**(1.0/q)

def g_1(K, p, q):
    return 1.0/p*K**(1.0/p-1)

def g_2(K, p, q):
    return 1.0/p*(1.0/p-1)*K**(1.0/p-2)

def h_0(K, m, N, p, q):
    return g_0(K, p, q) / IRR_0(K, m, N)

def h_1(K, m, N, p, q):
    return (IRR_0(K, m, N)*g_1(K, p, q) - g_0(K, p, q)*IRR_1(K, m, N)) / IRR_0(K, m, N)**2

def h_2(K, m, N, p, q):
    return ((IRR_0(K, m, N)*g_2(K, p, q) - IRR_2(K, m, N)*g_0(K, p, q) - 2.0*IRR_1(K, m, N)*g_1(K, p, q))/IRR_0(K, m, N)**2 \
            + 2.0*IRR_1(K, m, N)**2*g_0(K, p, q)/IRR_0(K, m, N)**3)

def option_pricer_1(F, iv, Tn, N, freq, p, q):
    D_T = libor_discount_curve(Tn)
    F_max = F*np.exp(4 * iv(F) * np.sqrt(Tn))
    # F_max = F + (4 * iv(F) * np.sqrt(Tn))/100
    # F_max = F+0.02
    I_rec = quad(lambda x: h_2(x, freq, N, p, q) * Black76Put(F, 
                                                              x, 
                                                              iv(x), 
                                                              Tn),
                 1E-10, F)[0]
    I_pay = quad(lambda x: h_2(x, freq, N, p, q) * Black76Call(F, 
                                                               x, 
                                                               iv(x), 
                                                               Tn),
                 F, F_max)[0]
    return D_T*g_0(F, p, q) + IRR_0(F, freq, N)*(I_rec + I_pay)

def option_pricer_2(F, iv, Tn, N, freq, p, q):
    D_T = libor_discount_curve(Tn)
    F_max = F*np.exp(4 * iv(F) * np.sqrt(Tn))
    x_star = 0.04**(p/q) # g(K) > 0 when K > x_star
    if x_star <= F:
        I_rec = quad(lambda x: h_2(x, freq, N, p, q) * Black76Put(F, 
                                                                  x, 
                                                                  iv(x),
                                                                  Tn),
                    x_star, F)[0]
        I_pay = quad(lambda x: h_2(x, freq, N, p, q) * Black76Call(F, 
                                                                   x, 
                                                                   iv(x),
                                                                   Tn),
                    F, F_max)[0]
        extra_term = h_1(x_star, freq, N, p, q) * Black76Put(F, 
                                                             x_star, 
                                                             iv(x_star),
                                                             Tn)
        return D_T*g_0(F, p, q) + IRR_0(F, freq, N)*(I_rec + I_pay + extra_term)
    else:
        I_pay = quad(lambda x: h_2(x, freq, N, p, q) * Black76Call(F, 
                                                                   x, 
                                                                   iv(x),
                                                                   Tn),
                    x_star, F_max)[0]
        extra_term = h_1(x_star, freq, N, p, q) * Black76Call(F, 
                                                              x_star, 
                                                              iv(x_star),
                                                              Tn)
        return IRR_0(F, freq, N)*(I_pay + extra_term)

# %%
# input
Tn = 5
N = 10
F = cms_rates_df[N][Tn]
alpha = sabr_alpha_df[N][Tn]
beta = 0.9
rho = sabr_rho_df[N][Tn]
nu = sabr_nu_df[N][Tn]
iv = lambda x: SABR(F, x, Tn, alpha, beta, rho, nu)

# output
pv1 = option_pricer_1(F, iv, 
                      Tn=5, N=10, freq=2, p=4, q=2)
pv2 = option_pricer_2(F, iv, 
                      Tn=5, N=10, freq=2, p=4, q=2)

print(f"""
==Input==
CMS 10y = {F}
SABR model parameters:
  alpha = {alpha}
  beta = {beta}
  rho = {rho}
  nu= {nu}
  
==Output==
Contract 1 PV = {pv1}
Contract 2 PV = {pv2}
""")