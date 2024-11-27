import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Parameters
initial_investment = 1000000  # Starting investment amount
years = 5
months = years * 12

# Monthly log returns and standard deviations for each industry
log_monthly_returns = np.array([
    0.008920762, 0.006226136, 0.004542421, 0.007491927,
    0.006620084, 0.007516075, 0.007238083, 0.006239284,
    0.007729881, 0.006490875
])
log_monthly_sd = np.array([0.06247597, 0.05248158, 0.04298716, 0.06150907, 0.05673731,
                           0.05714840, 0.06128117, 0.05447807, 0.06203335, 0.05163784])

# Weightings for high-risk portfolio
med_risk_weighting = np.array([0.4,-0.15,0.223449924,-0.015180713,-0.017866098,0.106999392,0.237423277,0.237860822,0.038247006,-0.06093261])

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
        random_returns = np.exp(np.random.multivariate_normal(log_monthly_returns, covariance_matrix))-1

        # Rebalance the weights monthly and calculate the weighted portfolio return
        weighted_returns = med_risk_weighting * random_returns
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
plt.xlabel('Years', fontsize=18)
plt.ylabel('Portfolio Value (Millions $)', fontsize=18)
plt.title('Projected Medium-Risk Portfolio Growth Over 5 Years (monthly rebalancing)', fontsize=18)

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
plt.legend(loc='upper left', title="Confidence Intervals", fontsize =18)
# Explicitly set font size for x and y labels

# Show plot
plt.show()

