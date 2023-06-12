
                #Answer 6 statistics-----

import pandas as pd

# Read the data file
data = pd.read_csv('path/to/your/data/file.csv')

# Explore the data
print(data.head())


from scipy.stats import shapiro

# Extract the "Blood Pressure Change" data
bp_change = data['Blood Pressure Change']

# Perform Shapiro-Wilk test
statistic, p_value = shapiro(bp_change)

# Interpret the results
alpha = 0.05
if p_value > alpha:
    print("The blood pressure change data follows a normal distribution (fail to reject H0)")
else:
    print("The blood pressure change data does not follow a normal distribution (reject H0)")
