import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
initial_investment = 1000000  # Starting investment amount
years = 5
months = years * 12

# Monthly log returns and standard deviations for each industry
log_monthly_returns = np.array([0.00446830, 0.01104980, 0.00811191, 0.00536067, 0.00582234,
                                0.00970914, 0.01022773, 0.00757651, 0.00699353, 0.00895203])
log_monthly_sd = np.array([0.06247597, 0.05248158, 0.04298716, 0.06150907, 0.05673731,
                           0.05714840, 0.06128117, 0.05447807, 0.06203335, 0.05163784])

# Weightings for high-risk portfolio
high_risk_weighting = np.array([-0.8784695, 0.926440937, 0.366425901, -0.66415044,
                                -0.11571517, 0.253262321, 0.14544411, 0.184731792,
                                -0.314544695, 1.096574744])

low_risk_weighting = np.array([0, 0.323600753, 0.493676024, 0, 0, 0, 0, 0, 0, 0.182723223])
# Simulation parameters
n_months = months
n_iterations = 10000

# Initialize an array to store portfolio values for each iteration over time
portfolio_values = np.ones((n_iterations, n_months + 1)) * initial_investment

# Simulate 10,000 paths
for i in range(n_iterations):
    portfolio_value = initial_investment  # Starting value
    for month in range(1, n_months + 1):
        # Generate random returns based on monthly log returns and standard deviations
        random_returns = np.exp(np.random.normal(log_monthly_returns, log_monthly_sd))-1

        # Rebalance the weights monthly and calculate the weighted portfolio return
        weighted_returns = high_risk_weighting * random_returns
        portfolio_monthly_return = np.sum(weighted_returns)

        # Update portfolio value
        portfolio_value *= (1+portfolio_monthly_return)
        portfolio_values[i, month] = portfolio_value

# Convert results to DataFrame for easier analysis
portfolio_paths_df = pd.DataFrame(portfolio_values).T

# Calculate percentiles for the fan chart
percentiles = [5, 10, 25, 50, 75, 90, 95]
percentile_data = portfolio_paths_df.quantile([p / 100 for p in percentiles], axis=1).T
percentile_data.columns = [f'{p} Percentile' for p in percentiles]

# Plot setup with Seaborn theme
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))

# Colors for confidence intervals (50%, 80%, 90%)
colors = sns.color_palette("Blues", 3)

# Fill between the 50%, 80%, and 90% confidence intervals symmetrically around the median
confidence_labels = ["90% Confidence Interval", "80% Confidence Interval", "50% Confidence Interval"]
interval_pairs = [(0, 6), (1, 5), (2, 4)]

for i, (lower, upper) in enumerate(interval_pairs):
    plt.fill_between(
        percentile_data.index,
        percentile_data.iloc[:, lower] / 1_000_000,       # Scale to millions
        percentile_data.iloc[:, upper] / 1_000_000,       # Scale to millions
        color=colors[i],
        alpha=0.3,
        label=confidence_labels[i]
    )

    # Add a border at the top and bottom of each filled interval
    plt.plot(
        percentile_data.index,
        percentile_data.iloc[:, lower] / 1_000_000,
        color='gray',
        linestyle=':',
        linewidth=1
    )
    plt.plot(
        percentile_data.index,
        percentile_data.iloc[:, upper] / 1_000_000,
        color='gray',
        linestyle=':',
        linewidth=1
    )

# Highlight the median line
plt.plot(percentile_data.index, percentile_data['50 Percentile'] / 1_000_000, color='blue', label='Median Outcome', linewidth=2)

# Set axis labels and title
plt.xlabel('Years', fontsize=12)
plt.ylabel('Portfolio Value (Millions $)', fontsize=12)
plt.title('Projected High-Risk Portfolio Growth Over 5 Years (monthly rebalancing)', fontsize=16)

# Set x-axis to display years instead of months
xticks_positions = np.arange(0, months + 1, 12)
xticks_labels = [f'{i}' for i in range(years + 1)]
plt.xticks(xticks_positions, xticks_labels)

# Adjust y-axis increments to increase by 1 million
plt.yticks(np.arange(0, percentile_data.max().max() / 1_000_000 + 1, 1))

# Adding secondary y-axis on the right
plt.gca().tick_params(axis='y', which='both', labelleft=True, labelright=True)

# Grid and legend
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
plt.legend(loc='upper left', title="Confidence Intervals")

# Show plot
plt.show()

