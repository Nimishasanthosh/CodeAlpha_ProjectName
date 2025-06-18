import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Load datasets
df1 = pd.read_csv(r"C:\Users\Admin\Desktop\unemployment_in_india\Unemployment in India.csv")
df2 = pd.read_csv(r"C:\Users\Admin\Desktop\unemployment_in_india\Unemployment_Rate_upto_11_2020.csv")

# Clean column names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Parse 'Date' column into datetime format
df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
df2 = df2.dropna(subset=['Date'])  # Drop rows where Date couldn't be parsed

# 1. National Unemployment Rate Over Time
plt.figure()
sns.lineplot(data=df2, x='Date', y='Estimated Unemployment Rate (%)', label='National Rate')
plt.axvspan(datetime(2020, 3, 1), datetime(2020, 8, 1), color='red', alpha=0.2, label='COVID-19 First Wave')
plt.title('Unemployment Rate Over Time (India)')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Explore regional/state-level trends
plt.figure()
sns.boxplot(data=df1, x='Region', y=' Estimated Unemployment Rate (%)')
plt.xticks(rotation=90)
plt.title('Unemployment Rate Distribution by Region')
plt.ylabel('Unemployment Rate (%)')
plt.tight_layout()
plt.show()

# 3. Monthly average trend to check seasonality
df2['Month'] = df2['Date'].dt.month
monthly_avg = df2.groupby('Month')['Estimated Unemployment Rate (%)'].mean()

plt.figure()
monthly_avg.plot(kind='bar', color='skyblue')
plt.title('Average Monthly Unemployment Rate (Seasonal Pattern)')
plt.xlabel('Month')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(range(0, 12), 
           ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.tight_layout()
plt.show()

# 4. Correlation with Employment Data
correlation_df = df1[['Estimated Employed', ' Estimated Unemployment Rate (%)']].dropna()
correlation = correlation_df.corr()

print("\nCorrelation Matrix:\n", correlation)

# 5. Insights
print("\nKey Insights:")
print("- Unemployment rate spiked during the initial months of COVID-19 (Mar–Jul 2020).")
print("- Some states show consistently higher unemployment rates—possible targets for policy.")
print("- A slight seasonality appears with increased unemployment around mid-year months.")
print("- Negative correlation between number of employed and unemployment rate, as expected.")

