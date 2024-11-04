import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define industries
industries = ['Hlth', 'MedEq', 'Drugs', 'Chems', 'PerSv', 'BusSv', 'Hardw', 'Boxes', 'Trans', 'Whlsl']

# Monthly log returns and standard deviations for each industry
log_monthly_returns = np.array([0.00446830, 0.01104980, 0.00811191, 0.00536067, 0.00582234,
                                0.00970914, 0.01022773, 0.00757651, 0.00699353, 0.00895203])
log_monthly_sd = np.array([0.06247597, 0.05248158, 0.04298716, 0.06150907, 0.05673731,
                           0.05714840, 0.06128117, 0.05447807, 0.06203335, 0.05163784])

low_risk_weighting = np.array([0, 0.323600753, 0.493676024, 0, 0, 0, 0, 0, 0, 0.182723223])

high_risk_weighting = np.array(
    [-0.8784695, 0.926440937, 0.366425901, -0.66415044, -0.11571517, 0.253262321, 0.14544411, 0.184731792, -0.314544695,
     1.096574744])

# Simulation parameters
initial_investment = 100  # Starting investment amount
years = 5
months = years * 12
iterations = 10000  # Number of simulations


# Function to simulate portfolio returns with different rebalancing strategies
def simulate_rebalancing(initial_weights):
    # Store results
    monthly_rebalance_returns = []
    annual_rebalance_returns = []
    no_rebalance_returns = []

    # Function to simulate monthly returns
    def simulate_monthly_returns():
        return np.exp(np.random.normal(log_monthly_returns, log_monthly_sd))-1

    # Monte Carlo simulation for each rebalancing strategy
    for _ in range(iterations):
        # Initial portfolio values
        portfolio_value_monthly = portfolio_value_annual = portfolio_value_no_rebalance = initial_investment
        weights_no_rebalance = initial_weights.copy()
        weights_annual = initial_weights.copy()

        for month in range(months):
            # Simulate monthly returns
            returns = simulate_monthly_returns()

            # Monthly Rebalancing: Reset weights every month to the initial weights
            portfolio_return_monthly = np.dot(initial_weights, returns)
            portfolio_value_monthly *= (1 + portfolio_return_monthly)

            # Annual Rebalancing: Reset weights every 12 months to the initial weights
            if month % 12 == 0 and month != 0:
                weights_annual = initial_weights.copy()
            portfolio_return_annual = np.dot(weights_annual, returns)
            portfolio_value_annual *= (1 + portfolio_return_annual)
            weights_annual *= (1 + returns)
            weights_annual /= weights_annual.sum()  # Normalize to maintain proportions

            # No Rebalancing: Let weights drift naturally
            portfolio_return_no_rebalance = np.dot(weights_no_rebalance, returns)
            portfolio_value_no_rebalance *= (1 + portfolio_return_no_rebalance)
            weights_no_rebalance *= (1 + returns)
            weights_no_rebalance /= weights_no_rebalance.sum()  # Normalize to maintain proportions

        # Record final portfolio values after 5 years for each strategy
        monthly_rebalance_returns.append(portfolio_value_monthly)
        annual_rebalance_returns.append(portfolio_value_annual)
        no_rebalance_returns.append(portfolio_value_no_rebalance)

    # Convert results to DataFrames for easier analysis
    results_df = pd.DataFrame({
        'Monthly Rebalance': monthly_rebalance_returns,
        'Annual Rebalance': annual_rebalance_returns,
        'No Rebalance': no_rebalance_returns
    })

    # Calculate summary statistics
    summary_stats = results_df.describe()

    # Calculate 5th, 10th, 90th, and 95th percentiles for each strategy
    percentiles = results_df.quantile([0.05, 0.10, 0.90, 0.95])

    # Plot the distributions of final portfolio values for each strategy
    plt.figure(figsize=(12, 8))
    sns.histplot(results_df['Monthly Rebalance'], label='Monthly Rebalance', kde=True)
    sns.histplot(results_df['Annual Rebalance'], label='Annual Rebalance', kde=True)
    sns.histplot(results_df['No Rebalance'], label='No Rebalance', kde=True)
    plt.xlabel("Final Portfolio Value ($)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Final Portfolio Value After 5 Years\n(10,000 Simulations per Strategy)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Display summary statistics and percentiles
    print("Summary Statistics for Final Portfolio Value After 5 Years:")
    print(summary_stats)
    print("\nPercentiles for Final Portfolio Value After 5 Years:")
    print(percentiles)


# Run simulation for low-risk weighting
print("Low-Risk Weighting Simulation:")
simulate_rebalancing(low_risk_weighting)

# Run simulation for high-risk weighting
print("\nHigh-Risk Weighting Simulation:")
simulate_rebalancing(high_risk_weighting)
