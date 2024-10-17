import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Create 'figures' directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Load the datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# Merge the datasets using the common 'ID' field
merged_df = pd.merge(pd.merge(df1, df2, on='ID'), df3, on='ID')

# Descriptive Statistical Analysis
# 1. Screen time summary (e.g., mean, median, mode)
screen_time_cols = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
print("Screen time summary (Weekdays vs Weekends):")
print(merged_df[screen_time_cols].describe())

# 2. Well-being summary statistics for all well-being indicators
well_being_cols = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 
                   'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
print("\nWell-being summary statistics:")
print(merged_df[well_being_cols].describe())

# Visualizations
# 1. Boxplot: Screen time on weekdays vs weekends across devices
plt.figure(figsize=(12, 6))
sns.boxplot(data=merged_df[screen_time_cols])
plt.title("Distribution of Screen Time (Weekdays vs Weekends)")
plt.xticks(rotation=45)
plt.savefig('figures/screen_time_boxplot.png')  # Save the boxplot
plt.close()  # Close the figure to free memory

# 2. Histogram for each well-being indicator
for col in well_being_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(merged_df[col], bins=5, kde=True)
    plt.title(f"Distribution of {col} Scores")
    plt.xlabel(f"{col} Score")
    plt.ylabel("Frequency")
    plt.savefig(f'figures/{col}_histogram.png')  # Save each histogram
    plt.close()  # Close the figure to free memory

# Inferential Statistical Analysis
# 1. Correlation between each screen time type and well-being
print("\nCorrelation Analysis:")
for screen_time in screen_time_cols:
    for well_being in well_being_cols:
        corr, _ = pearsonr(merged_df[screen_time], merged_df[well_being])
        print(f'Correlation between {screen_time} and {well_being}: {corr:.3f}')

# 2. T-test: Gender differences in all well-being indicators
print("\nT-test for gender differences in well-being:")
male_group = merged_df[merged_df['gender'] == 1]
female_group = merged_df[merged_df['gender'] == 0]
for well_being in well_being_cols:
    t_stat, p_value = ttest_ind(male_group[well_being], female_group[well_being])
    print(f'T-statistic for {well_being}: {t_stat:.3f}, P-value: {p_value:.3f}')

# 3. Predicting well-being scores based on screen time using Linear Regression
predicted_well_being = {}

for well_being in well_being_cols:
    y = merged_df[well_being]
    X = merged_df[screen_time_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    predicted_well_being[f'predicted_{well_being}'] = model.predict(X)
    score = model.score(X_test, y_test)
    print(f'R-squared for predicting {well_being}: {score:.3f}')

# Add predicted well-being scores to the dataframe
for key, value in predicted_well_being.items():
    merged_df[key] = value

# Define mental health condition based on predicted well-being scores
def define_mental_health(row, well_being_threshold=3):
    above_threshold = sum(row[f'predicted_{col}'] >= well_being_threshold for col in well_being_cols)
    below_threshold = len(well_being_cols) - above_threshold
    
    if above_threshold > (len(well_being_cols) / 2):
        return 'Good'
    elif below_threshold > (len(well_being_cols) / 2):
        return 'Poor'
    else:
        return 'Average'

merged_df['mental_health_condition'] = merged_df.apply(define_mental_health, axis=1)

# Save the result to a CSV file
merged_df.to_csv('mental_health_result.csv', index=False)

# Create a pie chart showing the percentage of mental health conditions
mental_health_counts = merged_df['mental_health_condition'].value_counts()

plt.figure(figsize=(8, 6))
plt.pie(mental_health_counts, labels=mental_health_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99','#ffcc99'])
plt.title('Percentage of Mental Health Conditions')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.savefig('figures/mental_health_pie_chart.png')  # Save the pie chart
plt.close()  # Close the figure to free memory
