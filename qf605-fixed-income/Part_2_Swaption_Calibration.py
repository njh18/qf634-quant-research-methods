# %%
import datetime as dt
import pandas as pd
import numpy as np
from math import log, sqrt, exp
from scipy.optimize import minimize, brentq, least_squares, fsolve
from scipy.stats import norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# %%
df_swap = pd.read_excel("data/IR Data.xlsx", sheet_name = 'Swaption' , engine = "openpyxl", skiprows = 2)
df_swap = df_swap.iloc[0:15,0:13]
df_swap.iloc[:,2:] = df_swap.iloc[:,2:]/100
df_swap = df_swap.set_index(['Expiry', 'Tenor'])

# # Calculate zero rates
disc_fac = pd.read_excel("data/IR Data.xlsx", sheet_name = 'OIS' , engine = "openpyxl", usecols = [0,1,2])
# Calculate discount factors
def convert_tenor(tenor):
    if "m" in tenor:
        return int(tenor.replace("m", "")) / 12  # Convert months to years
    elif "y" in tenor:
        return int(tenor.replace("y", ""))
    return None

disc_fac["Tenor_Years"] = disc_fac["Tenor"].apply(convert_tenor)
disc_fac["Discount_Factor"] = np.exp(-disc_fac["Rate"] * disc_fac["Tenor_Years"])
# print(disc_fac)

#interpolate for the 0.5y periods
rate_interp = interp1d(disc_fac["Tenor_Years"], disc_fac["Rate"], kind='linear', fill_value="extrapolate")
df_interp = interp1d(disc_fac["Tenor_Years"], disc_fac["Discount_Factor"], kind='linear', fill_value="extrapolate")
tenor_intervals = np.arange(0.5, 20.5, 0.5)  # 0.5, 1.0, 1.5, ..., 10.0

# Interpolate rates and discount factors for these intervals
interpolated_rates = rate_interp(tenor_intervals)
interpolated_dfs = df_interp(tenor_intervals)
# Create a DataFrame to store the results
df_results = pd.DataFrame({
    'Tenor': tenor_intervals,
    'Rate': interpolated_rates,
    'Discount_Factor': interpolated_dfs
})

# print(df_results)

# # Calculate forward swap rates
forward_swap_df = pd.read_excel('data/swap_rates.xlsx', sheet_name = 'swap_rates')\
# Drop the 'Unnamed: 0' column
forward_swap_df = forward_swap_df.drop(columns=['Unnamed: 0'])
# Rename columns
forward_swap_df.columns = ['1Y', '2Y', '3Y', '5Y', '10Y']
# Rename index
forward_swap_df.index = ['1Y', '5Y', '10Y']
forward_swap_df.index.name = "Expiry"
forward_swap_df.columns.name = "Tenor"
# print(forward_swap_df)

# %%
fwd_strikes = pd.read_excel("data/IR Data.xlsx", sheet_name = 'IRS' , engine = "openpyxl", usecols = [0,1,2])
# print(fwd_strikes)

# %%
#calculate the bps
df_swap_vals = df_swap.iloc[0:15,0:13]
bps = []
for col in df_swap_vals.columns[:]:
    col_stripped = col[:-3]  # Remove 'bps' suffix if present
    try:
        bps.append(float(col_stripped) / 10000)  # Convert to float
    except ValueError:
        bps.append(0)  # Assign 0 for non-numeric columns like 'ATM'
bps = np.array(bps)

# Black-Scholes formula
def blackscholescall(S, K, disc, sigma, T):
    d1 = (np.log(S / K) + (0.5 * sigma**2 * T)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    blackscholesprice = disc*(S * norm.cdf(d1) - K * norm.cdf(d2))
    return blackscholesprice

def blackscholesput(S, K, disc, sigma, T):
    d1 = (np.log(S / K) + (0.5 * sigma**2 * T)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    blackscholesprice = disc*(K * norm.cdf(-d2) - S * norm.cdf(-d1))
    return blackscholesprice

# Swaption price under Displaced-Diffusion model
def displaceddiffusioncall(S, K, disc, sigma, T, beta):
    adjusted_S = S / beta
    adjusted_K = K + ((1 - beta) / beta) * S
    adjusted_sigma = sigma * beta
    ddcallprice = blackscholescall(adjusted_S, adjusted_K, disc, adjusted_sigma, T)
    return ddcallprice

def displaceddiffusionput(S, K, disc, sigma, T, beta):
    adjusted_S = S / beta
    adjusted_K = K + ((1 - beta) / beta) * S
    adjusted_sigma = sigma * beta
    ddputprice = blackscholesput(adjusted_S, adjusted_K, disc, adjusted_sigma, T)
    return ddputprice

def ImpliedVolatility(S, K, disc, sigma, T, beta):
    """
    Calculate implied volatility based on market price

    Args:
    S(float): Forward price
    K(float): strike price
    T(float): time to expiration (in years)
    sigma(float): mid price between bid and offer

    Returns:
    Implied Volatility
    """
    if S <= K:
        price = displaceddiffusioncall(S, K, disc, sigma, T, beta)
        impliedVol = fsolve(lambda x: price -
                                    blackscholescall(S, K, disc, x, T),
                                    0.5)
    else:
        price = displaceddiffusionput(S, K, disc, sigma, T, beta)
        impliedVol = fsolve(lambda x: price -
                        blackscholesput(S, K, disc, x, T),
                            0.5)
    return impliedVol

# Calibration process
def DDcalibration(x, strikes, vols, S, disc, T):
    err = 0.0
    sigma = vols[5]
    for i, vol in enumerate(vols):
        implied_vol= ImpliedVolatility(S, strikes[i], disc, sigma, T, x)
        err += (vol - implied_vol)**2
    return err

# %%
expiries = ['1Y', '5Y', '10Y']
tenors = ['1Y', '2Y', '3Y', '5Y', '10Y']
DD_Sigma=pd.DataFrame(np.zeros((3,5)), index=expiries,columns=tenors)
DD_Beta=pd.DataFrame(np.zeros((3,5)), index=expiries,columns=tenors)

# %%
# print(df_swap)

# %%
# Function to calculate PVBP
def calculate_pvbp(expiry_int, tenor_int, df_results):
    # Generate the payment periods (every 6 months)
    payment_periods = np.arange(expiry_int, expiry_int + tenor_int + 0.5, 0.5)

    # Interpolate discount factors for the payment periods
    discount_factors = np.interp(payment_periods, df_results["Tenor"], df_results["Discount_Factor"])

    # Calculate PVBP
    pvbp = 0.5 * 0.01 * np.sum(discount_factors)
    return pvbp

# DD Calibration
beta_initial_guess = [0.3]
for expiry in expiries:
    for tenor in tenors:
        swap_forward = forward_swap_df.loc[expiry,tenor]
        market_vol = df_swap.loc[(expiry, tenor), :].values
        strikes = swap_forward + bps
        expiry_int = int(expiry.strip()[:-1])
        tenor_int = int(tenor.strip()[:-1])
        print(tenor_int)

        # Calculate PVBP
        disc_factor = calculate_pvbp(expiry_int, tenor_int, df_results)

        res = least_squares(
            lambda beta: DDcalibration(
                beta,
                strikes,
                market_vol,
                swap_forward,
                disc_factor,
                T= expiry_int
            ),
            beta_initial_guess,
            bounds=(0,2)
        )
        if res.success:
            DD_Sigma.loc[expiry, tenor] = market_vol[5]
            DD_Beta.loc[expiry, tenor] = res.x[0]
            print(f"Success: DD_Sigma[{tenor}][{expiry}] = {market_vol[5]}, DD_Beta[{tenor}][{expiry}] = {res.x[0]}")
        else:
            print(f"Optimization failed for expiry={expiry}, tenor={tenor}")

# %%
print(DD_Sigma)

# %%
print(DD_Beta)

# %%
for expiry in expiries:
    for tenor in tenors:
        swap_forward = forward_swap_df.loc[expiry,tenor]
        market_vol = df_swap.loc[(expiry, tenor), :].values
        strikes = swap_forward + bps
        expiry_int = int(expiry.strip()[:-1])
        tenor_int = int(tenor.strip()[:-1])
        disc_factor = 0.5* sum(df_results.Discount_Factor[2*expiry_int:(expiry_int+tenor_int)*2])
        T= expiry_int

        dd_vols_swap = [ImpliedVolatility(swap_forward, i, disc_factor, DD_Sigma.loc[expiry,tenor], T, DD_Beta.loc[expiry,tenor])
                        for i in strikes]


        plt.figure(figsize=(8,6))
        plt.scatter(strikes, market_vol, marker = 's')
        plt.plot(strikes, dd_vols_swap, '--r')

# SABR
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

def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    beta=0.9
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], beta, x[1], x[2]))**2

    return err

# %%
# SABR Calibration
expiries = ['1Y', '5Y', '10Y']
tenors = ['1Y', '2Y', '3Y', '5Y', '10Y']
sabr_alpha=pd.DataFrame(np.zeros((3,5)), index=expiries,columns=tenors)
sabr_rho=pd.DataFrame(np.zeros((3,5)), index=expiries,columns=tenors)
sabr_nu=pd.DataFrame(np.zeros((3,5)), index=expiries,columns=tenors)

# %%
sabr_initial_guess = [0.15, -0.6, 1.3]

for expiry in expiries:
    for tenor in tenors:
        swap_forward = forward_swap_df.loc[expiry,tenor]
        market_vol = df_swap.loc[(expiry, tenor), :].values
        strikes = swap_forward + bps

        expiry_int = int(expiry.strip()[:-1])

        res = least_squares(
            lambda x: sabrcalibration(
                x,
                strikes,
                market_vol,
                swap_forward,
                T= expiry_int
            ),
            sabr_initial_guess,
             bounds=([0.1,-0.75,0.5], [0.2,-0.3, 2.3]),
            xtol=1e-6,  # Looser stopping condition
            ftol=1e-6
        )
        if res.success:
            sabr_alpha.loc[expiry, tenor] = res.x[0]
            sabr_rho.loc[expiry, tenor] = res.x[1]
            sabr_nu.loc[expiry, tenor] = res.x[2]
            print(f"Success: sabr_alpha[{tenor}][{expiry}] = {res.x[0]}, sabr_rho[{tenor}][{expiry}] = {res.x[1]}, sabr_nu[{tenor}][{expiry}] = {res.x[2]}")
        else:
            print(f"Optimization failed for expiry={expiry}, tenor={tenor}")
            print(f"Final error: {sabrcalibration(res.x, strikes, market_vol, swap_forward, T=expiry_int)}")
            print(f"Final parameters: {res.x}")


# %%
print(sabr_alpha)

# %%
print(sabr_rho)

# %%
print(sabr_nu)

# %%
#plot diagrams for DD and SABR
num_plots = len(expiries) * len(tenors)  # 3 * 5 = 15 plots
rows = (num_plots + 1) // 2

fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(15, rows * 5))  # Create a 3x5 grid of subplots
fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust spacing between plots

axes = axes.flatten()

plot_index = 0

for expiry in expiries:
    for tenor in tenors:
        swap_forward = forward_swap_df.loc[expiry,tenor]
        market_vol = df_swap.loc[(expiry, tenor), :].values
        strikes = swap_forward + bps

        expiry_int = int(expiry.strip()[:-1])
        tenor_int = int(tenor.strip()[:-1])

        disc_factor = 0.5* sum(df_results.Discount_Factor[2*expiry_int:(expiry_int+tenor_int)*2])
        T= expiry_int

        beta = 0.9
        dd_vols_swap = [ImpliedVolatility(swap_forward, i, disc_factor, DD_Sigma.loc[expiry,tenor], T, DD_Beta.loc[expiry,tenor])
                        for i in strikes]

        # def SABR(F, K, T, alpha, beta, rho, nu):
        sabr_vol = [SABR(swap_forward, i, T, sabr_alpha.loc[expiry, tenor], beta, sabr_rho.loc[expiry, tenor], sabr_nu.loc[expiry, tenor])
                   for i in strikes]

        ax = axes[plot_index]
        ax.scatter(strikes, market_vol, marker = 's', label = 'market obs')
        ax.plot(strikes, dd_vols_swap, '--r', label = 'displaced diffusion')
        ax.plot(strikes, sabr_vol, '--g', label = 'SABR')
        ax.set_title(f"Expiry {expiry}- Tenor {tenor}")
        ax.legend()
        ax.set_xlabel("Strikes")
        ax.set_ylabel("Volatility")

        plot_index += 1

output_path = "data/sabr_calibrated.xlsx"

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    sabr_alpha.to_excel(writer, sheet_name='sabr_alpha')
    sabr_rho.to_excel(writer, sheet_name='sabr_rho')
    sabr_nu.to_excel(writer, sheet_name='sabr_nu')

# Pricing Swaptions
# Function to interpolate the 10Y tenor from 1Y to 5Y expiry
def interpolate_10y_tenor(df):
    # Extract the 10Y column
    tenors = df.index
    values = df['10Y']

    # Convert the tenors to numerical values for interpolation
    x = np.array([1, 5, 10])  # Corresponding to 1Y, 5Y, 10Y
    y = values.values

    # Create a linear interpolation function
    interp_func = np.interp

    # Generate the interpolated values for 1Y to 10Y
    new_x = np.arange(1, 10.1, 1)  # From 1Y to 10Y in steps of 1Y
    interp_values = interp_func(new_x, x, y)

    # Create a new DataFrame for the interpolated values
    interp_df = pd.DataFrame(interp_values, index=new_x, columns=['10Y'])

    return interp_df

# Apply the function to the various dataFrames
sigma_int = interpolate_10y_tenor(DD_Sigma)
# print(sigma_int)
beta_int = interpolate_10y_tenor(DD_Beta)
# print(beta_int)
alpha_int = interpolate_10y_tenor(sabr_alpha)
# print(alpha_int)
rho_int = interpolate_10y_tenor(sabr_rho)
# print(rho_int)
nu_int = interpolate_10y_tenor(sabr_nu)
# print(nu_int)

# %%
# Function to interpolate the 10Y tenor from 1Y to 5Y expiry
def interpolate_10y_tenor(df):
    # Extract the 10Y column
    expiries = df.index
    values = df['10Y']

    # Convert the expiries to numerical values for interpolation
    x = np.array([1, 5, 10])  # Corresponding to 1Y, 5Y, 10Y
    y = values.values

    # Create a linear interpolation function
    interp_func = np.interp

    # Generate the interpolated values for 1Y to 10Y
    new_x = np.arange(1, 10.1, 1)  # From 1Y to 10Y in steps of 1Y
    interp_values = interp_func(new_x, x, y)

    # Create a new DataFrame for the interpolated values
    interp_df = pd.DataFrame(interp_values, index=new_x, columns=['10Y'])
    interp_df.index.name = 'Expiry'

    return interp_df

# Apply the function to the forward_swap_df DataFrame
forward_swapint = interpolate_10y_tenor(forward_swap_df)
# print(forward_swapint)
strikes=np.arange(0.01,0.081,0.01)
T = 2
expiry_int = 2
tenor_int = 10
two_year_expiry_val = forward_swapint.loc[2.0, '10Y']
sigma_int_val = sigma_int.loc[2.0, '10Y']
beta_int_val = beta_int.loc[2.0, '10Y']
alpha_int_val = alpha_int.loc[2.0, '10Y']
rho_int_val = rho_int.loc[2.0, '10Y']
nu_int_val = nu_int.loc[2.0, '10Y']
disc1 = 0.5 * sum(df_results.Discount_Factor[2*expiry_int:(expiry_int+tenor_int)*2])
# print(disc1)

payer_dd = [displaceddiffusioncall(two_year_expiry_val, i, disc1, sigma_int_val, T, beta_int_val) for i in strikes]
# print(payer_dd)
payer_SABR=[blackscholescall(two_year_expiry_val, i, disc1, SABR(two_year_expiry_val, i, T, alpha_int_val,0.9 ,rho_int_val, nu_int_val), T) for i in strikes]
# print(payer_SABR)
df_payer = pd.DataFrame({'Strikes':strikes, 'Displaced Diffusion':payer_dd, 'SABR':payer_SABR})
print(df_payer)

# 8X10 year
strikes=np.arange(0.01,0.081,0.01)
T = 8
expiry_int = 8
tenor_int = 10
eight_year_expiry_val = forward_swapint.loc[8.0, '10Y']
# print(eight_year_expiry_val)
sigma_int_val = sigma_int.loc[8.0, '10Y']
beta_int_val = beta_int.loc[8.0, '10Y']
alpha_int_val = alpha_int.loc[8.0, '10Y']
rho_int_val = rho_int.loc[8.0, '10Y']
nu_int_val = nu_int.loc[8.0, '10Y']
disc1 = 0.5 * sum(df_results.Discount_Factor[2*expiry_int:(expiry_int+tenor_int)*2])
# print(disc1)

receiver_dd = [displaceddiffusionput(eight_year_expiry_val, i, disc1, sigma_int_val, T, beta_int_val) for i in strikes]
# print(receiver_dd)
receiver_SABR=[blackscholesput(eight_year_expiry_val, i, disc1, SABR(eight_year_expiry_val, i, T, alpha_int_val,0.9 ,rho_int_val, nu_int_val), T) for i in strikes]
# print(receiver_SABR)
df_receiver = pd.DataFrame({'Strikes':strikes, 'Displaced Diffusion':receiver_dd, 'SABR':receiver_SABR})
print(df_receiver)


