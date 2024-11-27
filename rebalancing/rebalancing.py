import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define industries
industries = ['Hlth', 'MedEq', 'Drugs', 'Chems', 'PerSv', 'BusSv', 'Hardw', 'Boxes', 'Trans', 'Whlsl']

# Monthly log returns and standard deviations for each industry
log_monthly_returns = np.array([
    0.008920762, 0.006226136, 0.004542421, 0.007491927,
    0.006620084, 0.007516075, 0.007238083, 0.006239284,
    0.007729881, 0.006490875
])

log_monthly_sd = np.array([0.06247597, 0.05248158, 0.04298716, 0.06150907, 0.05673731,
                           0.05714840, 0.06128117, 0.05447807, 0.06203335, 0.05163784])
covariance_matrix = [
    [0.003903246, 0.002685027, 0.001772876, 0.002608728, 0.002472589, 0.00273905, 0.002181002, 0.002034591, 0.002680744, 0.0024383],
    [0.002685027, 0.002754317, 0.001673374, 0.00216643, 0.001942309, 0.0024147, 0.001961979, 0.001852768, 0.002253361, 0.001943873],
    [0.001772876, 0.001673374, 0.001847896, 0.001577198, 0.001373, 0.001578189, 0.001564952, 0.001263841, 0.001506237, 0.001416454],
    [0.002608728, 0.00216643, 0.001577198, 0.003783366, 0.002654266, 0.002790747, 0.00260694, 0.002615968, 0.003116348, 0.002704951],
    [0.002472589, 0.001942309, 0.001373, 0.002654266, 0.003219122, 0.002657204, 0.002242111, 0.002135759, 0.002876172, 0.002377889],
    [0.00273905, 0.0024147, 0.001578189, 0.002790747, 0.002657204, 0.00326594, 0.002440937, 0.002343735, 0.002914485, 0.002439491],
    [0.002181002, 0.001961979, 0.001564952, 0.00260694, 0.002242111, 0.002440937, 0.003755381, 0.001993078, 0.002719649, 0.002308104],
    [0.002034591, 0.001852768, 0.001263841, 0.002615968, 0.002135759, 0.002343735, 0.001993078, 0.00296786, 0.002565307, 0.002095886],
    [0.002680744, 0.002253361, 0.001506237, 0.003116348, 0.002876172, 0.002914485, 0.002719649, 0.002565307, 0.003848137, 0.002710357],
    [0.0024383, 0.001943873, 0.001416454, 0.002704951, 0.002377889, 0.002439491, 0.002308104, 0.002095886, 0.002710357, 0.002666467]
]

low_risk_weighting = np.array([
    0.0000000000, 0.3333332062, 0.3333333793, 0.0070522549,
    0.0629399444, 0.0268897994, 0.0588509138, 0.0986484668,
    0.0000000000, 0.0789520352
])

med_risk_weighting = np.array([0.4,-0.15,0.223449924,-0.015180713,-0.017866098,0.106999392,0.237423277,0.237860822,0.038247006,-0.06093261])
high_risk_weighting = np.array([0.4,-0.048690155,-0.23441178,0.246928697,-0.136708383,0.4,0.334766089,0.013374716,0.255801721,-0.231060905])

# Simulation parameters
initial_investment = 100  # Starting investment amount
years = 5
months = years * 12
iterations = 10000 # Number of simulations


# Function to simulate portfolio returns with different rebalancing strategies
def simulate_rebalancing(initial_weights):
    # Store results
    monthly_rebalance_returns = []
    annual_rebalance_returns = []
    no_rebalance_returns = []

    # Function to simulate monthly returns
    def simulate_monthly_returns():
        return np.exp(np.random.multivariate_normal(log_monthly_returns, covariance_matrix)) -1

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

    prob_loss = (results_df < initial_investment).mean()
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
    print(prob_loss)



# Run simulation for low-risk weighting
print("Low-Risk Weighting Simulation:")
simulate_rebalancing(low_risk_weighting)

# Run simulation for high-risk weighting
print("\nHigh-Risk Weighting Simulation:")
simulate_rebalancing(high_risk_weighting)

# Run simulation for high-risk weighting
print("\nMedium-Risk Weighting Simulation:")
simulate_rebalancing(med_risk_weighting)
