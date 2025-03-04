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
    print (df_cons.head())
    print(df_cons.tail())

    return df_cons

############
###Swap segment ###
#############

def get_swap_data (file_path):
    
    #step 1: data cleaning
    df_swap = pd.read_excel(file_path, sheet_name = 'SofrCurve' , engine = "openpyxl")
    #extract out relevant zero curves, 1Y - 10 Y
    df_swap1 = df_swap.iloc[list(range(6, 16))]
    #print(df_swap1)

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

    ### Step 3: Calculating the PV01
    # Iterate over the rows (for each tenor)
    absolute_returns = []
    # Iterate over each row (tenor) in the DataFrame
    # for index, row in df_swap1.iterrows():
    #     # Extract the 'Tenor' column value
    #     tenor = row['Tenor']
        
    #     # Get the values of the current row (excluding the 'Tenor' and 'T' columns)
    #     row_values = row.iloc[2:].values  # This should contain the values from the date columns
        
    #     # Calculate the daily absolute return for the tenor by taking the difference between consecutive values
    #     daily_returns = [row_values[i] - row_values[i-1] for i in range(1, len(row_values))]

    #     daily_returns.insert(0, 0)  # The first day should have 0

    #     # Add the calculated daily returns as a new row for the current tenor
    #     # We add 'Tenor' and 'T' columns, while repeating the daily return values for each date
    #     new_row = [f"Daily Return ({tenor})"] + daily_returns
    #     absolute_returns.append(new_row)
    # Extract relevant tenors (1Y to 10Y)

    # Calculate absolute daily returns using .diff()
    df_abs_ret = df_swap1.iloc[:, 2:].diff(axis=1)

    # Fill the first column with 0 (since there's no previous value to subtract from)
    df_abs_ret = df_abs_ret.dropna(axis=1, how='all')

    # Add a 'Tenor' column to identify the rows
    df_abs_ret.insert(0, 'Tenor', df_swap1['Tenor'])

    # Rename columns to indicate that these are absolute returns
    print(df_abs_ret)
    # Create a DataFrame from the new rows
    # Append the new rows to the original DataFrame

    return df_swap1, value_10y, df_abs_ret, pv_swap

#finally we combine everything###
def get_var(df_cons, df_swap1, value_10y, df_abs_ret, pv_swap):

    ##Load stock data and calculate

    ##calculate the metrics
    cov_mat = df_cons.cov()

    appl_mean = np.average(df_cons['Return_AAPL'])
    #print(appl_mean, appl_std)
    msft_mean = np.average(df_cons['Return_MSFT'])
    f_mean = np.average(df_cons['Return_F'])
    bac_mean = np.average(df_cons['Return_BAC'])

    #calculate the weights and L for each stock 
    mean_vec = np.array([appl_mean, msft_mean, f_mean, bac_mean])
    #print(mean_vec)

    # # Select the last row and extract only the 'Adj Close' columns
    # last_row_adj_close = df_merged.iloc[-1, 1:5].tolist()
    # #print(last_row_adj_close)

    # appl_p0 = last_row_adj_close[0]
    # msft_p0 = last_row_adj_close[1]
    # f_p0 = last_row_adj_close[2]
    # bac_p0 = last_row_adj_close [3]

    ###Calculate swap data
    #Step 1: calculate mean first for the swaps
    # Extract the 10Y daily absolute return for all dates
    daily_returns_10y = df_abs_ret[df_abs_ret['Tenor'] == '10Y'].iloc[:, 1:].values.flatten()
    # Calculate the mean of the absolute daily returns for 10 Y tenor
    mean_absolute_daily_returns = np.mean(daily_returns_10y) 
    print (f"the mean for 10Y swap is : {mean_absolute_daily_returns}")
    # Store the mean value in the existing NumPy array
    mean_vec = np.append(mean_vec, mean_absolute_daily_returns)

    print(mean_vec)

    #Step 2: Calculate the swap P&L
    # ##first term of L(swap)
    # pv01_10y = (math.exp(-10*(value_10y+0.0001))-math.exp(-10*value_10y)) / 0.0001 
    # delta_abs10y =  df_swap1.iloc[19, 2:].sum()  # Exclude 'Tenor' and 'T' columns
    # print(f"the delta abs 10year value is: {delta_abs10y}")

    # #second term of L(swap)
    # # Extract zero-coupon rates (Z^q) for the last date
    # zero_coupon_rates = df_swap1.iloc[:10, -1].values
    # zero_coupon_rates = np.array(zero_coupon_rates, dtype=float)

    # # Extract daily returns (ΔZ^q_0) for all dates and sum them for each tenor
    # daily_returns = df_swap1.iloc[10:, 2:].sum(axis=1).values
    # daily_returns = np.array(daily_returns, dtype=float) 

    # # Tenors (T^q)
    # tenors = df_swap1.iloc[:10, 1].values
    # tenors = np.array(tenors, dtype=float)

    # # Calculate PV01^q for each tenor
    # pv01 = (np.exp(-tenors * (zero_coupon_rates + 0.0001)) - np.exp(-tenors * zero_coupon_rates)) / 0.0001

    # # Multiply PV01^q by ΔZ^q_0 and sum the results
    # result = np.sum(pv01 * daily_returns)

    # #print(result)
    # #Calculating final L(swap)
    # l_swap = (-notional_swap * pv01_10y * delta_abs10y) - (notional_swap * strike_rate * result)
    # print(f"The P&L of the swap is {l_swap}")

    ##Construct combined dataframe of stock relative returns + swap absolute return
    #Step 1: Manipulate the abs return to merge with stock relative returns
    # Transpose df_abs_ret
    daily_return_10y = df_abs_ret[df_abs_ret['Tenor'] == '10Y']
    daily_return_10y = daily_return_10y.set_index('Tenor').T

    # Reset the index to make dates a column
    daily_return_10y = daily_return_10y.reset_index().rename(columns={'index': 'Date'})

    # Convert the 'Date' column to datetime
    daily_return_10y['Date'] = pd.to_datetime(daily_return_10y['Date'])

    # Ensure df_cons has a 'Date' column aligned with new_rows_df
    # Assuming df_cons has the same dates as new_rows_df
    df_cons['Date'] = daily_return_10y['Date']

    # Merge the two dataframes on the 'Date' column
    merged_df = pd.merge(df_cons, daily_return_10y, on='Date', how='inner')
    # Print the merged dataframe

    merged_df = merged_df.drop(columns=['Date'])
    print(merged_df.tail(10))

    #Step 2: calculate the cov of the combined df
    # ##calculate the metrics
    cov_mat = merged_df.cov()
    print(cov_mat)

    #calculate the w_parametric
    w_parametric = np.array([notional_stock, notional_stock, notional_stock, notional_stock, pv_swap])
    print(w_parametric)
    m1d =  np.inner(w_parametric, mean_vec)
    # print(m1d)
    v1d = np.inner(np.dot(w_parametric, cov_mat), w_parametric)
    var1d = np.abs(stat.norm.ppf(0.05, loc=m1d, scale=np.sqrt(v1d))) #provide 5 percentile with mean as input
    
    return var1d

def main():
    df_cons = get_stock_data(file_path)
    df_swap1 = get_swap_data(file_path)[0]
    value_10y = get_swap_data(file_path)[1]
    df_abs_ret = get_swap_data(file_path)[2]
    pv_swap = get_swap_data(file_path)[3]
    var1d = get_var(df_cons, df_swap1, value_10y, df_abs_ret, pv_swap)
    print(f"1 day parametric var (95%) is {var1d}")

if __name__ == "__main__":
    main()