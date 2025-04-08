"""
Authors: Jun Hao & Lynn
Description: Convexity Correction 
1. Using calibrated SABR model to value constant maturity swap (CMS) products
2. Compare forward swap rates with CMS rates
"""

import pandas as pd
import numpy as np
import os
import pickle
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import least_squares
from datetime import datetime
import matplotlib.pyplot as plt

## read OIS excel sheet 
file_path = os.path.abspath("data/IR Data.xlsx")
df_ois = pd.read_excel(file_path, sheet_name="OIS", usecols=[0,1,2])

# convert tenor to years 
tenor_mapping = {
    '6m': 0.5,
    '1y': 1,
    '2y': 2,
    '3y': 3,
    '4y': 4,
    '5y': 5,
    '7y': 7,
    '10y': 10,
    '15y': 15,
    '20y': 20,
    '30y': 30
}

df_ois['Tenor'] = df_ois['Tenor'].map(tenor_mapping)
df_ois['Tenor_Delta'] = df_ois['Tenor'].diff().fillna(0.5)

## read libor discount curve & ois pckl files 
libor_curve = pickle.load(open("data/libor_discount_curve.pkl", "rb"))
ois_discount_curve = pickle.load(open("data/ois_discount_curve.pkl", "rb"))

## read forward swap rates 
file_path_swap = os.path.abspath("data/swap_rates.xlsx")
forward_swap_rates = pd.read_excel(file_path_swap)
forward_swap_rates.rename(columns={'Unnamed: 0':'Expiry'}, inplace=True)
forward_swap_rates = forward_swap_rates.set_index('Expiry')
forward_swap_rates.columns.name = 'Tenor'

r = np.arange(0.5, 20.5, 0.5)
interpolated_libors = libor_curve(r)
interpolated_ois = ois_discount_curve(r)
interpolated_ois_df = pd.DataFrame({
    'Tenor': r,
    'OIS_Discount_Factor': interpolated_ois,
    'Forward_Libor_Rates': interpolated_libors
    })

interpolated_ois_df = interpolated_ois_df.set_index('Tenor')

## daily compounded overnight rate 
def calc_compounded_on_rate(f_t, delta_T):
    """
    calculate the daily compounded return based on the overnight rate

    Args:
    f_t(float): overnight rate
    T(float): tenor

    Returns:
    compounded overnight rate
    """
    return 1 / ((1 + f_t/360) ** (360 * delta_T))

## discount factor 
def calc_discount_factor(prev_df, f_t, delta_T):
    """
    calculate the discount factor for a given tenor

    Args:
    prev_df(float): previous discount factor
    f_t(float): overnight rate
    delta_T(float): time difference

    Returns:
    discount factor
    """
    return prev_df * calc_compounded_on_rate(f_t, delta_T)

## forward libor rate 
def calc_forward_libor(prev_df, current_df, delta_T):
    """
    calculate the forward libor rate for a given tenor

    Args:
    prev_df(float): previous discount factor
    current_df(float): current discount factor
    delta_T(float): time difference

    Returns:
    forward libor rate
    """
    return (1/delta_T) *(prev_df/current_df - 1)

## black 76 model 
class black_76_model:
    def __init__(self, F: float, K: float, sigma: float, discount_factor: float, T: float):
        self.F = F
        self.K = K
        self.sigma = sigma
        self.discount_factor = discount_factor
        self.T = T
        self.d1 = self.calc_black_scholes_d1()
        self.d2 = self.calc_black_scholes_d2()

    def calc_black_scholes_d1(self) -> float:
        sigma_sqrt_time = self.sigma * np.sqrt(self.T)
        return (np.log(self.F / self.K) + (np.power(self.sigma, 2)/2) * self.T ) / sigma_sqrt_time
    
    def calc_black_scholes_d2(self) -> float:
        sigma_sqrt_time = self.sigma * np.sqrt(self.T)
        return self.d1 - sigma_sqrt_time
    
    def blackscholes_call(self):
        return self.discount_factor*(self.F*norm.cdf(self.d1) - self.K*norm.cdf(self.d2))

    def blackscholes_put(self):
        return self.discount_factor*(self.F*norm.cdf(-self.d1) + self.K*norm.cdf(-self.d2))
    
## SABR model
def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    # if K is at-the-money-forward
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

## read sabr params data and pre-process
file_path_sabr = os.path.abspath("data/sabr_calibrated.xlsx")
df_sabr_alpha = pd.read_excel(file_path_sabr, sheet_name="sabr_alpha")
df_sabr_rho = pd.read_excel(file_path_sabr, sheet_name="sabr_rho")
df_sabr_nu = pd.read_excel(file_path_sabr, sheet_name="sabr_nu")

def prep_sabr_params(df):
    df = df.rename(columns = {'Unnamed: 0': 'Expiry'})
    df.set_index('Expiry', inplace=True)
    df.columns.name = 'Tenors'
    return df

df_sabr_alpha = prep_sabr_params(df_sabr_alpha)
df_sabr_rho = prep_sabr_params(df_sabr_rho)
df_sabr_nu = prep_sabr_params(df_sabr_nu)

print("=============== df_sabr_alpha ====================")
print(df_sabr_alpha)
print("=============== df_sabr_rho =============== ")
print(df_sabr_rho)
print("===============df_sabr_nu =============== ")
print(df_sabr_nu)

## interpolation of sabr params 
expiry_range = np.arange(0.25, 10.25, 0.25) # quarterly expiry range

def sabr_params_interp(df):
    interpolated_values = {}

    for tenor in df.columns:
        interp = interpolate.interp1d(
            df.index.str.replace('Y', '').astype(float), 
            df[tenor].astype(float),
            kind='linear', 
            fill_value='extrapolate'
            )
    
        interpolated_values[tenor] = interp(expiry_range)
    return pd.DataFrame(interpolated_values, index=expiry_range)


df_sabr_alpha_interp = sabr_params_interp(df_sabr_alpha)
df_sabr_rho_interp = sabr_params_interp(df_sabr_rho)
df_sabr_nu_interp = sabr_params_interp(df_sabr_nu)

## CMS rates calculation
def IRR_0(K, m, N):
    """
    Implementation of IRR(K) function 

    Args:
    K(float): strike rate
    m(float): tenor
    N(float): number of periods
    """
    value = 1/K * ( 1.0 - 1/(1 + K/m)**(N*m) )
    return value

def IRR_1(K, m, N):
    """
    Implementation of IRR'(K) function (1st derivative)
    """
    firstDerivative = -1/K*IRR_0(K, m, N) + 1/(K*m)*N*m/(1+K/m)**(N*m+1)
    return firstDerivative

def IRR_2(K, m, N):
    """ 
    Implementation of IRR''(K) function (2nd derivative)
    """
    secondDerivative = -2/K*IRR_1(K, m, N) - 1/(K*m*m)*(N*m)*(N*m+1)/(1+K/m)**(N*m+2)
    return secondDerivative


def g_0(K):
    """ 
    Implementation of g(K) function
    """
    return K

def g_1(K):
    """
    Implementation of g'(K) function (1st derivative)
    """
    return 1.0

def g_2(K):
    """
    Implementation of g''(K) function (2nd derivative)
    """
    return 0.0

def h_0(K, m, N):
    """ 
    Implementation of h(K) function
    """
    value = g_0(K) / IRR_0(K, m, N)
    return value

def h_1(K, m, N):
    """ 
    Implementation of h'(K) function (1st derivative)
    """
    firstDerivative = (IRR_0(K, m, N)*g_1(K) - g_0(K)*IRR_1(K, m, N)) / IRR_0(K, m, N)**2
    return firstDerivative

def h_2(K, m, N):
    """ 
    Implementation of h''(K) function (2nd derivative)
    """
    secondDerivative = ((IRR_0(K, m, N)*g_2(K) - IRR_2(K, m, N)*g_0(K) - 2.0*IRR_1(K, m, N)*g_1(K))/IRR_0(K, m, N)**2 
                        + 2.0*IRR_1(K, m, N)**2*g_0(K)/IRR_0(K, m, N)**3)
    return secondDerivative

## IRR settled option price 
def irr_settled_option_price(F, discount_factor, K, sigma, T, m, N, swap_type):
    """
    Calculate the price of an IRR-settled swaption using the Black-76 model

    Parameters:
        F (float): forward swap rate
        discount_factor (float): discount factor
        K (float): strike
        sigma (float): volatility
        T (float): time to expiry
        m (int): number of payments per year
        N (int): Tenor
        swap_type (str): Type of swap (Payer or Receiver).

    Returns:
        float: Option price.
    """
    irr_0 = IRR_0(F, m, N)
    df_numeraire = 1  # discount factor D(t,T) = 1
    black76_model = black_76_model(F, K, sigma, df_numeraire, T)
    if swap_type == 'payer':
        option_price = black76_model.blackscholes_call()
    elif swap_type == 'receiver':
        option_price = black76_model.blackscholes_put()
    else: 
        raise NameError("Invalid swap type.")
    return discount_factor * irr_0 * option_price

Expiry = [1, 5, 10]
Tenors = [1, 2, 3, 5, 10]
cms_rate = pd.DataFrame(np.zeros((len(Expiry), len(Tenors))), index=Expiry, columns=Tenors)

## interpolation of forward swap rates
expiry_range = np.arange(0.25, 10.25, 0.25) # quarterly expiry range

def interploate_df(df):
    interpolated_values = {}

    for tenor in df.columns:
        interp = interpolate.interp1d(
            df.index,
            df[tenor].astype(float),
            kind='linear', 
            fill_value='extrapolate'
            )
    
        interpolated_values[tenor] = interp(expiry_range)
    return pd.DataFrame(interpolated_values, index=expiry_range)

forward_swap_df_interp = interploate_df(forward_swap_rates)

## Interpoation of OIS discount factionrs
full_range = np.arange(0.25,20.5,0.25)

interp_ois = interpolate.interp1d(
    interpolated_ois_df.index,
    interpolated_ois_df["OIS_Discount_Factor"],
    kind='linear',
    fill_value='extrapolate'
)

new_interp_ois_df = pd.DataFrame({'T': interp_ois(full_range)}, index=full_range)


## CMS rates calculation
def calc_cms_rate(F, m, N, T):
    alpha = df_sabr_alpha_interp.loc[T, f"{N}Y"]
    beta = 0.9
    rho = df_sabr_rho_interp.loc[T, f"{N}Y"]
    nu = df_sabr_nu_interp.loc[T, f"{N}Y"]  
    discount_factor = new_interp_ois_df.loc[T]["T"]

    atm_vol = SABR(F,F,T,alpha, beta, rho, nu)
    maxK = F * np.exp(4 * atm_vol * np.sqrt(T))
    integrand_receive = quad(lambda k: h_2(k, m, N)*irr_settled_option_price(F, 
                                                                                discount_factor,
                                                                                k,
                                                                                SABR(F,k,T,alpha, beta, rho, nu),
                                                                                T, 
                                                                                m, 
                                                                                N, 
                                                                                'receiver'), 
                                                                                1e-6, # lower bound close to 0
                                                                                F, 
                                                                                limit=100
                                                                                )
                                                                                
    integrand_pay = quad(lambda k: h_2(k, m, N)*irr_settled_option_price(F, 
                                                                            discount_factor, 
                                                                            k, 
                                                                            SABR(F,k,T,alpha, beta, rho, nu),
                                                                            T, 
                                                                            m, 
                                                                            N, 
                                                                            'payer'), 
                                                                            F, 
                                                                            maxK, # upper bound instead of np.inf
                                                                            limit=100 # subintervals
                                                                            )
    return g_0(F) + integrand_pay[0] + integrand_receive[0]

for expiry in Expiry:
    for tenor in Tenors:
        F = forward_swap_rates.loc[expiry, tenor]
        T = expiry
        m = 2
        N = tenor        
        cms_rate.loc[expiry, tenor] = calc_cms_rate(F, T, m, N)

cms_rate.columns.name = 'Tenor'
cms_rate.index.name = 'Expiry'
cms_rate = cms_rate.reset_index()
cms_rate = cms_rate.set_index('Expiry')


## calculation of PV of CMS leg
# PV of a leg receiving CMS10y semi-annually over the next 5 years
pv = 0

for start in np.arange(0.5, 5, 0.5):
    disc_factor = interpolated_ois_df.loc[start]["OIS_Discount_Factor"]
    F = forward_swap_df_interp.loc[start, 10]
    T = start
    m = 2
    N = 10
    current_cms_rate = calc_cms_rate(F, T, m, N)
    pv += disc_factor * 0.5 * current_cms_rate
    print(f"start={start}, df={disc_factor}, cms_rate={current_cms_rate}, pv={pv}")

print(f"\n final PV={pv}")

## PV of a leg receiving CMS2y quarterly over the next 10 years
pv = 0

for start in np.arange(0.25, 10, 0.25):
    disc_factor = new_interp_ois_df.loc[start]["T"]
    F = forward_swap_df_interp.loc[start, 2]
    T = start
    m = 2
    N = 2
    current_cms_rate = calc_cms_rate(F, T, m, N)
    pv += disc_factor * 0.5 * current_cms_rate
    print(f"start={start}, df={disc_factor}, cms_rate={current_cms_rate}, pv={pv}")

print(f"\n final PV={pv}")

## comparison of forward swap rates and CMS rates
## plot cms rates with forward swap rates for diff expiries 
expiries = [1,5,10]
tenors = [1,2,3,5,10]

for expiry in expiries: 
    plt.figure(figsize=(10, 6))
    plt.plot(cms_rate.columns, cms_rate.loc[expiry], label=f'CMS Rates')
    plt.plot(forward_swap_rates.columns, forward_swap_rates.loc[expiry], label=f'Forward Swap Rates', linestyle='--')
    plt.title(f'CMS Rates vs Forward Swap Rates for {expiry}Y Expiry')
    plt.xlabel('Tenors (Years)')
    plt.ylabel('Rates')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()