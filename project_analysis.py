
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind

# Load the datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# Merge the datasets using the common 'ID' field
merged_df = pd.merge(pd.merge(df1, df2, on='ID'), df3, on='ID')

# Descriptive Statistical Analysis
# 1. Screen time summary (e.g., mean, median, mode)
print("Screen time summary (Weekdays vs Weekends):")
screen_time_cols = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
print(merged_df[screen_time_cols].describe())

# 2. Well-being summary (e.g., mean for all well-being indicators)
print("\nWell-being summary statistics:")
well_being_cols = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 
                   'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
print(merged_df[well_being_cols].describe())

# Visualizations
# 1. Boxplot: Screen time on weekdays vs weekends across devices
plt.figure(figsize=(10, 6))
sns.boxplot(data=merged_df[screen_time_cols])
plt.title("Distribution of Screen Time (Weekdays vs Weekends)")
plt.xticks(rotation=45)
plt.show()

# 2. Histogram: Distribution of optimism scores
plt.figure(figsize=(8, 5))
sns.histplot(merged_df['Optm'], bins=5, kde=True)
plt.title("Distribution of Optimism Scores")
plt.xlabel("Optimism Score")
plt.ylabel("Frequency")
plt.show()

# Inferential Statistical Analysis
# 1. Correlation between screen time and optimism
print("\nCorrelation Analysis:")
corr, _ = pearsonr(merged_df['S_wk'], merged_df['Optm'])
print(f'Correlation between weekday smartphone use and optimism: {corr:.3f}')

# 2. T-test: Gender differences in well-being (Optimism score)
print("\nT-test for gender differences in optimism:")
male_group = merged_df[merged_df['gender'] == 1]['Optm']
female_group = merged_df[merged_df['gender'] == 0]['Optm']
t_stat, p_value = ttest_ind(male_group, female_group)
print(f'T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}')

# Additional Inferential Analysis
# 3. T-test: Minority status differences in well-being (Cheerfulness score)
print("\nT-test for minority status differences in cheerfulness:")
majority_group = merged_df[merged_df['minority'] == 0]['Cheer']
minority_group = merged_df[merged_df['minority'] == 1]['Cheer']
t_stat_minority, p_value_minority = ttest_ind(majority_group, minority_group)
print(f'T-statistic: {t_stat_minority:.3f}, P-value: {p_value_minority:.3f}')

# 4. Correlation between weekend video game use and feeling confident
print("\nCorrelation between weekend video game use and feeling confident:")
corr_conf, _ = pearsonr(merged_df['G_we'], merged_df['Conf'])
print(f'Correlation between weekend video game use and confidence: {corr_conf:.3f}')
