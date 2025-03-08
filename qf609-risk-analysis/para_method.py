import pandas as pd 
import numpy as np
import scipy.interpolate
import statistics
import scipy.stats as stat
import math

# File path
file_path = "hist_data.xlsm"
notional_stock = 1000000
notional_swap = 100000000
strike_rate = 0.042

############
###Stock segment ###
#############

#Process stock data
def get_stock_data(file_path):
    sheet_names = pd.ExcelFile(file_path).sheet_names
    first_sheet = sheet_names[0]  # The sheet to exclude
    dfs = {}
    
    # Read each sheet except the first one
    for sheet in sheet_names[1:]:  # Skipping the first sheet
        dfs[sheet] = pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")
        dfs[sheet].rename(columns={"Adj Close": f"Adj Close_{sheet}"}, inplace=True)  # Rename column for clarity

    # Merge all remaining sheets on "Date"
    df_merged = list(dfs.values())[0]  # Start with the first DataFrame (after skipping the first sheet)

    for sheet, df in list(dfs.items())[1:]:
        df_merged = pd.merge(df_merged, df, on="Date", how="inner") 


    # Convert Date column to datetime format
    df_merged["Date"] = pd.to_datetime(df_merged["Date"])

    # Calculate relative return for each stock
    for col in df_merged.columns:
        if "Adj Close" in col:  # Apply only to price columns
            df_merged[f"Return_{col.split('_')[-1]}"] = df_merged[col].pct_change()

    #drop NaN
    df_merged = df_merged.dropna()

    df_cons = df_merged.iloc[:,5:]

    ##calculate the metrics
    #print(df_cons)
    cov_mat = df_cons.cov()
    #print(cov_mat)
    appl_mean = np.average(df_cons['Return_AAPL'])
    #print(appl_mean, appl_std)
    msft_mean = np.average(df_cons['Return_MSFT'])
    f_mean = np.average(df_cons['Return_F'])
    bac_mean = np.average(df_cons['Return_BAC'])

    #calculate the weights and L for each stock 
    mean_vec_stock = np.array([appl_mean, msft_mean, f_mean, bac_mean])
    #print(mean_vec_stock)

    #calculate the w_parametric
    w_parametric_stock = np.array([notional_stock, notional_stock, notional_stock, notional_stock])
    #print(w_parametric_stock)
    m1d_stock =  np.inner(w_parametric_stock, mean_vec_stock)
    # print(m1d)
    v1d_stock = np.inner(np.dot(w_parametric_stock, cov_mat), w_parametric_stock)
    #var1d_stock = np.abs(stat.norm.ppf(0.05, loc=m1d_stock, scale=np.sqrt(v1d_stock))) #provide 5 percentile with mean as input

    return m1d_stock,v1d_stock

############
###Swap segment ###
#############

def get_swap_data (file_path):
    
    #step 1: data cleaning
    df_swap = pd.read_excel(file_path, sheet_name = 'SofrCurve' , engine = "openpyxl")
    #extract out relevant zero curves, 1Y - 10 Y
    df_swap1 = df_swap.iloc[list(range(6, 16))]
    #print(df_swap1)

    ###extract out zero_rates as a df
    zero_rates = df_swap1.iloc[:, [0, -1]]
    #print(zero_rates)

    #convert column names to strings
    zero_rates.columns = zero_rates.columns.astype(str)
    #print(zero_rates.columns)
    # calculate pv fixed
    fix_sum = 0 
    #Iterate through the DataFrame
    for index, row in zero_rates.iterrows():
        # Get the tenor and zero rate
        tenor = row['Tenor']
        zero_rate = row['2023-10-30 00:00:00']
        
        # Convert tenor to numerical value (e.g., "1Y" to 1)
        tenor_years = int(tenor[:-1])  # Remove 'Y' and convert to integer
        
        # Multiply tenor in years by zero rate and add to total_sum
        fix_sum += math.exp(-tenor_years * zero_rate)

    #step 2: Calculate PV of swap
    # print("Total sum:", fix_sum)
    pv_fixed = notional_swap * strike_rate * fix_sum
    # Filter the DataFrame to get the row for '10Y'
    tenor_10y_row = zero_rates[zero_rates['Tenor'] == '10Y']

    # Extract the value for 2023-10-30 00:00:00 from the filtered row
    value_10y = tenor_10y_row['2023-10-30 00:00:00'].iloc[0]

    flt_rate = math.exp(-value_10y*10)
    pv_float = notional_swap * (1-flt_rate)
    #print(pv_float)
    #calculate the pv of the swap
    pv_swap = pv_float - pv_fixed
    print(f"The PV of the swap is {pv_swap}.")


    #####calculate key metrics now######

    # Calculate absolute daily returns using .diff()
    df_abs_ret = df_swap1.iloc[:, 2:].diff(axis=1)

    # Fill the first column with 0 (since there's no previous value to subtract from)
    df_abs_ret = df_abs_ret.dropna(axis=1, how='all')

    # Add a 'Tenor' column to identify the rows
    df_abs_ret.insert(0, 'Tenor', df_swap1['Tenor'])

    mean_values = df_abs_ret.iloc[:, 1:].mean(axis=1)
    df_mean = pd.DataFrame({
    'Tenor': df_abs_ret['Tenor'],
    'Mean_Daily_Value': mean_values
    })
    #print(df_mean)
    
    df_comb= pd.merge(zero_rates,df_mean,on='Tenor', how='inner')
    df_comb.rename(columns={'2023-10-30 00:00:00': 'Zero_Rate'}, inplace=True)
    df_comb['Tenor'] = df_comb['Tenor'].str.replace('Y', '').astype(float)
    df_comb['pv01'] = (np.exp(-df_comb['Tenor'] * (df_comb['Zero_Rate'] + 0.0001)) - np.exp(-df_comb['Tenor'] * df_comb['Zero_Rate'])) * 10000
    df_comb['w_parametric'] = -notional_swap*strike_rate*df_comb['pv01']
    adjustment = notional_swap * df_comb['pv01'].iloc[-1]

    # Update the last value in the 'w_parametric' column
    df_comb['w_parametric'].iloc[-1] = df_comb['w_parametric'].iloc[-1] - adjustment
    #print(df_comb)

    #calculate the cov matrix of these Tenors
    df_abs_ret.set_index('Tenor', inplace=True)
    df_abs_ret_trans = df_abs_ret.T
    # Calculate the covariance matrix
    cov_mat_swap = df_abs_ret_trans.cov()
    #print(cov_mat_swap)

    #calculate the w_parametric
    w_parametric_swap = df_comb['w_parametric'].to_numpy()
    mean_vec_swap = df_comb['Mean_Daily_Value'].to_numpy()
    #print(w_parametric_stock)
    m1d_swap =  np.inner(w_parametric_swap, mean_vec_swap)
    # print(m1d)
    v1d_swap = np.inner(np.dot(w_parametric_swap, cov_mat_swap), w_parametric_swap)
    #var1d_swap = np.abs(stat.norm.ppf(0.05, loc=m1d_swap, scale=np.sqrt(v1d_swap))) 
    return m1d_swap, v1d_swap

def main():
    stock_var = get_stock_data(file_path)
    m1d_stock = stock_var [0]
    v1d_stock = stock_var [1]
    swap_var = get_swap_data(file_path)
    m1d_swap = swap_var[0]
    v1d_swap = swap_var[1]
    var1d_comb = np.abs(stat.norm.ppf(0.05, loc=(m1d_stock+m1d_swap), scale=np.sqrt(v1d_stock+v1d_swap)))
    print(f"1 day parametric var (95%) for SWAP is {var1d_comb}")
if __name__ == "__main__":
    main()