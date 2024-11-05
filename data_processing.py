import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

def process_data(dacha_file, population_file):
    # Load the data
    df = pd.read_csv(dacha_file)
    fd = pd.read_csv(population_file)
    
    # Merge the datasets on 'City'
    df = df.merge(fd, on='City', how='left')
    
    # Log-transform the variables
    df['log_Units_Sold'] = np.log(df['Units Sold'])
    df['log_Experts'] = np.log(df['Experts'])
    df['log_Staff'] = np.log(df['Staff'])
    df['log_Population'] = np.log(df['Population'])
    
    # Dictionary for month names
    month_names = {
        1: 'January',
        2: 'February',
        3: 'March',
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July',
        8: 'August',
        9: 'September',
        10: 'October',
        11: 'November',
        12: 'December'
    }

    # Create binary indicator columns for each month
    for month_num, month_name in month_names.items():
        df[month_name] = (df['Month'] == month_num).astype(int)
    
    return df


def calculate_recommended_staffing(df, W_E, W_S, pop_estimate, med_inc_estimate, Contribution_Margin=100, Rev=1000):
    # Create a monthly recommended employment dataframe
    recdf = pd.DataFrame(columns=['Month', 'RecExperts', 'RecStaff', 'Quantity', 'Revenue', 'GrossProfit', 'EmpWages'])
    
    # Log-transform the relevant columns
    df['log_Units_Sold'] = np.log(df['Units Sold'])
    df['log_Experts'] = np.log(df['Experts'])
    df['log_Staff'] = np.log(df['Staff'])
    df['log_Population'] = np.log(df['Population'])
    df['log_Med_Inc'] = np.log(df['Med Inc'])

    # Regression model (LEAVE OUT JANUARY)
    X = df[['log_Experts', 'log_Staff', 'log_Population', 'log_Med_Inc',
            'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December']]
    X = sm.add_constant(X)
    y = df['log_Units_Sold']
    model = sm.OLS(y, X).fit()

    # Extract parameters
    alpha = model.params['log_Experts']
    beta = model.params['log_Staff']
    gamma = model.params['log_Population']
    delta = model.params['log_Med_Inc']
    feb = model.params['February']
    mar = model.params['March']
    apr = model.params['April']
    may = model.params['May']
    jun = model.params['June']
    jul = model.params['July']
    aug = model.params['August']
    sep = model.params['September']
    oct = model.params['October']
    nov = model.params['November']
    dec = model.params['December']

    # Total Factor Productivity (TFP)
    A = np.exp(model.params['const'])
    
    # Create month-to-month contributions dictionary
    month_contributions = {
        1: 1,
        2: np.exp(feb),
        3: np.exp(mar),
        4: np.exp(apr),
        5: np.exp(may),
        6: np.exp(jun),
        7: np.exp(jul),
        8: np.exp(aug),
        9: np.exp(sep),
        10: np.exp(oct),
        11: np.exp(nov),
        12: np.exp(dec),
    }

    # Loop through the months and calculate the optimal staffing levels
    for month_num in range(1, 13):
        month = month_contributions[month_num]
        
        # Adjusted TFP with Population and Median Income inputs
        adjusted_A = A * (pop_estimate ** gamma) * (med_inc_estimate ** delta) * month

        # Objective function to minimize marginal productivity difference
        def objective(x):
            Experts, Staff = x

            # Marginal Productivity of Experts
            EMargProd = Contribution_Margin * adjusted_A * alpha * (Experts ** (alpha - 1)) * (Staff ** beta)

            # Marginal Productivity of Staff
            SMargProd = Contribution_Margin * adjusted_A * (Experts ** alpha) * (beta * Staff ** (beta - 1))

            # Differences from wages
            ExpertDiff = EMargProd - W_E
            StaffDiff = SMargProd - W_S

            # Return the sum of squared differences
            return ExpertDiff ** 2 + StaffDiff ** 2

        # Optimize the number of Experts and Staff
        initial_guesses = [1, 1]  # Initial guess for Experts and Staff
        result = minimize(objective, initial_guesses, bounds=((0, None), (0, None)))

        # Optimal values
        optimal_experts, optimal_staff = result.x

        # Recalculate the marginal productivities at optimal values
        optimal_EMargProd = Contribution_Margin * adjusted_A * alpha * (optimal_experts ** (alpha - 1)) * (optimal_staff ** beta)
        optimal_SMargProd = Contribution_Margin * adjusted_A * (optimal_experts ** alpha) * (beta * optimal_staff ** (beta - 1))

        # Calculate Quantity and other metrics
        Q = (adjusted_A * (optimal_experts ** alpha) * (optimal_staff ** beta))
        Revenue = Rev * Q
        GrossProfit = Contribution_Margin * Q
        EmpWages = (optimal_experts * W_E) + (optimal_staff * W_S)

        # Append the results for the current month
        recdf.loc[len(recdf)] = [month_num, optimal_experts, optimal_staff, Q, Revenue, GrossProfit, EmpWages]

    return recdf

def calculate_net_income(df, fixed_cost = 500_000):
    gross_profit = df.GrossProfit.sum()
    wages = df.EmpWages.sum()
    
    net_income = gross_profit - wages - fixed_cost
    print(net_income)