



import pandas as pd

data = pd.read_csv('path/to/your/data/file.csv')

print(data.head())


dispersion_before = data['Blood Pressure Before'].max() - data['Blood Pressure Before'].min()
dispersion_after = data['Blood Pressure After'].max() - data['Blood Pressure After'].min()

variance_before = data['Blood Pressure Before'].var()
variance_after = data['Blood Pressure After'].var()

std_deviation_before = data['Blood Pressure Before'].std()
std_deviation_after = data['Blood Pressure After'].std()


print("Dispersion (Range) - Blood Pressure Before:", dispersion_before)
print("Dispersion (Range) - Blood Pressure After:", dispersion_after)

print("Variance - Blood Pressure Before:", variance_before)
print("Variance - Blood Pressure After:", variance_after)

print("Standard Deviation - Blood Pressure Before:", std_deviation_before)
print("Standard Deviation - Blood Pressure After:", std_deviation_after)


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Calculate mean and confidence interval
mean_before = data['Blood Pressure Before'].mean()
mean_after = data['Blood Pressure After'].mean()

confidence_interval_before = stats.norm.interval(0.95, loc=mean_before, scale=std_deviation_before / np.sqrt(len(data)))
confidence_interval_after = stats.norm.interval(0.95, loc=mean_after, scale=std_deviation_after / np.sqrt(len(data)))

# Plot the means
plt.bar(['Before', 'After'], [mean_before, mean_after], yerr=[mean_before-confidence_interval_before[0], mean_after-confidence_interval_after[0]])
plt.title('Mean Blood Pressure')
plt.ylabel('Mean')
plt.show()


mad_before = data['Blood Pressure Before'].mad()
mad_after = data['Blood Pressure After'].mad()

# Interpret the results
print("Mean Absolute Deviation - Blood Pressure









