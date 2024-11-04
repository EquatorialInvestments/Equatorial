import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Simulation parameters
initial_investment = 1_000_000  # Starting investment amount in dollars
monthly_return = 0.009216122   # Expected monthly return
monthly_volatility = 0.0430   # Monthly volatility (risk) as a standard deviation
years = 5                       # Forecasting period (5 years)
months = years * 12             # Total months for simulation
iterations = 10000              # Number of different simulated scenarios

# Monte Carlo simulation
portfolio_paths = []
for _ in range(iterations):
    portfolio_values = [initial_investment]
    for month in range(months):
        monthly_growth_rate = np.random.normal(monthly_return, monthly_volatility)
        portfolio_values.append(portfolio_values[-1] *np.exp(monthly_growth_rate))
    portfolio_paths.append(portfolio_values)

# Convert results to a DataFrame for easier analysis
portfolio_paths_df = pd.DataFrame(portfolio_paths).T

# Calculate percentiles
percentiles = [5, 10, 25, 50, 75, 90, 95]
percentile_data = portfolio_paths_df.quantile([p / 100 for p in percentiles], axis=1).T
percentile_data.columns = [f'{p} Percentile' for p in percentiles]

# Plot setup with symmetric colors around the median
plt.figure(figsize=(12, 8))
colors = sns.color_palette("Blues", len(percentiles) // 2 + 1)

# Fill between each percentile range symmetrically around the median
confidence_labels = ["90% Confidence Interval", "80% Confidence Interval", "50% Confidence Interval"]
for i in range(len(percentiles) // 2):
    plt.fill_between(
        percentile_data.index,
        percentile_data.iloc[:, i] / 1_000_000,       # Scale to millions
        percentile_data.iloc[:, -i-1] / 1_000_000,    # Scale to millions
        color=colors[i],
        alpha=0.5,
        label=confidence_labels[i]
    )

    # Add a dotted border at the top and bottom of each filled interval
    plt.plot(
        percentile_data.index,
        percentile_data.iloc[:, i] / 1_000_000,
        color='gray',
        linestyle=':',
        linewidth=1
    )
    plt.plot(
        percentile_data.index,
        percentile_data.iloc[:, -i-1] / 1_000_000,
        color='gray',
        linestyle=':',
        linewidth=1
    )

# Highlight the median line
plt.plot(percentile_data.index, percentile_data['50 Percentile'] / 1_000_000, color='blue', label='Median Outcome', linewidth=2)

# Set axis labels and title
plt.xlabel('Years', fontsize=12)
plt.ylabel('Portfolio Value (Millions $)', fontsize=12)
plt.title('Projected Low-Risk Portfolio Growth Over 5 Years', fontsize=16)

# Set x-axis to display years instead of months
xticks_positions = np.arange(0, months + 1, 12)
xticks_labels = [f' {i}' for i in range(years + 1)]
plt.xticks(xticks_positions, xticks_labels)

# Adding secondary y-axis on the right
plt.gca().tick_params(axis='y', which='both', labelleft=True, labelright=True)


# Grid and legend
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
plt.legend(loc='upper left', title="Confidence Intervals")

# Show plot
plt.show()
