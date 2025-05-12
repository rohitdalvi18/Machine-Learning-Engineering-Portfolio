# Import the core libraries we'll use throughout our analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Mount Google Drive
from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)

project_folder = '/content/drive/My Drive/Colab Notebooks'

file_path = os.path.join(project_folder, "GlobalTemperatures.csv")

df = pd.read_csv(file_path)

# dimensions of the dataframe
print(df.shape)

df.head(5)

# display the last 5 rows
df.tail(5)

# look at the data structure & data types
df.info()

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

# Summary statistics
df.describe()

# Boxplot for Monthly Anomalies
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["Monthly_Anomaly"])
plt.title("Boxplot of Monthly Temperature Anomalies")
plt.show()


# Histogram of Monthly Anomalies
plt.figure(figsize=(10, 5))
sns.histplot(df["Monthly_Anomaly"], bins=50, kde=True)
plt.title("Distribution of Monthly Temperature Anomalies")
plt.xlabel("Temperature Anomaly (°C)")
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x=df["Year"], y=df["Monthly_Anomaly"], label="Monthly Anomalies", alpha=0.5)
sns.lineplot(x=df["Year"], y=df["Five_Year_Anomaly"], label="5-Year Average", color="red")
plt.axhline(y=0, color="black", linestyle="--", alpha=0.6)
plt.title("Trend of Monthly Temperature Anomalies Over Time")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.legend()
plt.show()

# Years with max temperature anomalies

max_anomaly = df.loc[df["Monthly_Anomaly"].idxmax()]

print(f"Highest temperature anomaly: {max_anomaly['Year']} ({max_anomaly['Monthly_Anomaly']}°C)")


# Years with min temperature anomalies

min_anomaly = df.loc[df["Monthly_Anomaly"].idxmin()]

print(f"Lowest temperature anomaly: {min_anomaly['Year']} ({min_anomaly['Monthly_Anomaly']}°C)")

# Group by decade and compute mean anomalies
df["Decade"] = (df["Year"] // 10) * 10
decade_trend = df.groupby("Decade")["Monthly_Anomaly"].mean()

# Plot trend over decades
plt.figure(figsize=(12, 6))
sns.lineplot(x=decade_trend.index, y=decade_trend.values, marker="o", color="red")
plt.title("Average Temperature Anomalies per Decade (1900-2020)")
plt.xlabel("Decade")
plt.ylabel("Temperature Anomaly (°C)")
plt.show()

# Averaging different periods
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["Annual_Anomaly"], label="Annual", alpha=0.5)
plt.plot(df["Year"], df["Five_Year_Anomaly"], label="5-Year Avg", linestyle="--")
plt.plot(df["Year"], df["Ten_Year_Anomaly"], label="10-Year Avg", linestyle="-.")
plt.plot(df["Year"], df["Twenty_Year_Anomaly"], label="20-Year Avg", linestyle=":")
plt.title("Temperature Anomalies with Different Averaging Periods")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.legend()
plt.show()

# Group by month and compute mean anomalies
monthly_avg = df.groupby("Month")["Monthly_Anomaly"].mean()

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, palette="coolwarm")
plt.title("Average Monthly Temperature Anomalies")
plt.xlabel("Month")
plt.ylabel("Temperature Anomaly (°C)")
plt.xticks(range(1, 13))
plt.show()

# Comparing Pre-1937 and Post-1937

df["Period"] = np.where(df["Year"] < 1937, "Pre-1937", "Post-1937")
monthly_comparison = df.groupby(["Month", "Period"])["Monthly_Anomaly"].mean().unstack()

# Plot
monthly_comparison.plot(kind="bar", figsize=(12, 6))
plt.title("Monthly Anomalies Before and After 1937")
plt.xlabel("Month")
plt.ylabel("Temperature Anomaly (°C)")
plt.xticks(range(12), labels=range(1, 13), rotation=0)
plt.legend(title="Period")
plt.show()


# Group by month and compute mean and standard deviation of anomalies
monthly_trends = df.groupby("Month")["Monthly_Anomaly"].agg(["mean", "std"])

# Plot mean anomalies for each month
plt.figure(figsize=(10, 5))
sns.lineplot(x=monthly_trends.index, y=monthly_trends["mean"], marker="o", color="red", label="Mean Anomaly")
plt.fill_between(monthly_trends.index,
                 monthly_trends["mean"] - monthly_trends["std"],
                 monthly_trends["mean"] + monthly_trends["std"],
                 color="red", alpha=0.2, label="Std Dev Range")
plt.title("Monthly Temperature Anomalies (Mean & Variability)")
plt.xlabel("Month")
plt.ylabel("Temperature Anomaly (°C)")
plt.xticks(range(1, 13))
plt.legend()
plt.show()


# Define the baseline period (1951-1980)
baseline_period = df[(df["Year"] >= 1951) & (df["Year"] <= 1980)]

# Compute mean and standard deviation for baseline period
baseline_monthly = baseline_period.groupby("Month")["Monthly_Anomaly"].agg(["mean", "std"])

# Compute the deviation from the baseline for all years
df["Deviation_from_Baseline"] = df.apply(lambda row: row["Monthly_Anomaly"] - baseline_monthly.loc[row["Month"], "mean"], axis=1)

# Group by month to analyze deviation statistics
monthly_deviation = df.groupby("Month")["Deviation_from_Baseline"].agg(["mean", "std"])

# Plot deviation from baseline
plt.figure(figsize=(10, 5))
sns.barplot(x=monthly_deviation.index, y=monthly_deviation["mean"], palette="coolwarm")
plt.errorbar(monthly_deviation.index, monthly_deviation["mean"], yerr=monthly_deviation["std"], fmt="o", color="black", capsize=5)
plt.title("Monthly Temperature Anomalies Compared to 1951-1980 Baseline")
plt.xlabel("Month")
plt.ylabel("Deviation from Baseline (°C)")
plt.xticks(range(1, 13))
plt.show()


# Compute the mean and standard deviation
mean_anomaly = df["Monthly_Anomaly"].mean()
std_anomaly = df["Monthly_Anomaly"].std()

# Binning
def categorize_anomaly(anomaly):
    if anomaly <= mean_anomaly - 2 * std_anomaly:
        return "Extreme Cold"
    elif anomaly <= mean_anomaly - std_anomaly:
        return "Cold"
    elif anomaly < mean_anomaly + std_anomaly:
        return "Neutral"
    elif anomaly < mean_anomaly + 2 * std_anomaly:
        return "Warm"
    else:
        return "Extreme Warm"

df["Anomaly_Category"] = df["Monthly_Anomaly"].apply(categorize_anomaly)

# Display category counts
print(df["Anomaly_Category"].value_counts())


# Group by Year and count occurrences of each category
category_trend = df.groupby(["Year", "Anomaly_Category"])["Monthly_Anomaly"].count().unstack()

# Plot category frequencies over time
plt.figure(figsize=(12, 6))
category_trend.plot(kind="area", stacked=True, colormap="coolwarm", alpha=0.7)
plt.title("Frequency of Temperature Anomaly Categories Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Months")
plt.legend(title="Anomaly Category")
plt.show()

