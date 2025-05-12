import pandas as pd             # data manipulation
import numpy as np              # numerical operation
import matplotlib.pyplot as plt #basic plotting
import seaborn as sns           #statistical visulaization

sns.set(color_codes=True)       # allows for use of shorthand color codes

# mount Google Drive
from google.colab import drive
import os

#Mount Google Drive
drive.mount('/content/drive', force_remount=True)

#Specify folder you want to access
project_folder = '/content/drive/My Drive/Colab Notebooks'

file_path = os.path.join(project_folder, "Titanic-Dataset.csv")

df = pd.read_csv(file_path)

# dimensions of the dataframe
print(df.shape)

# display the top  rows
df.head(5)

# display the last 5 rows
df.tail(5)

min_pclass = df['Pclass'].min()
max_pclass = df['Pclass'].max()
print(f"Min passenger class: {min_pclass}")
print(f"Max passenger class: {max_pclass}")

# look at the data structure
# What are the data types
df.info()

# Basic statistical summary
print(df.describe())

# create a boxplot for age
plt.figure(figsize=(8,6)) #adjust figure size
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Boxplot of Age by Survival')
plt.xlabel('Survived (0=  No, 1 = Yes)')
plt.ylabel('Age')
plt.show()

plt.figure(figsize=(8,6)) #adjust figure size
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Boxplot of Fare by Survival')
plt.xlabel('Survived (0=  No, 1 = Yes)')
plt.ylabel('Fare')
plt.show()

def generate_bar_chart(data, column):

  plt.figure(figsize=(10,6)) #specify dimensions of visualization

  if column in ["Fare", "Age"]:
    sns.histplot(data[column], bins=30, kde=True)
    plt.title(f"Distribution of {column}", fontsize=16)
    plt.xlabel(column, fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)

  else:
    sns.countplot(x=column, data=data)
    plt.title(f"Count frequncy of {column}", fontsize=16)
    plt.xlabel(column, fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)

  plt.tight_layout()
  plt.show()

columns_to_plot = ['Survived', 'Pclass', 'Sex', 'Fare', 'Age', 'SibSp', 'Parch']

for column in columns_to_plot:
  generate_bar_chart(df, column)

sns.pairplot(df, vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
plt.show()

sns.pairplot(df, hue='Survived', vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
plt.show()

num_null_age = df['Age'].isnull().sum()
print(f"Number of null values in Age: {num_null_age}")

median_age_by_group = df.groupby(['Sex', 'Pclass'])['Age'].median()
print(median_age_by_group)

def fill_age(row):
  if pd.isnull(row['Age']):
    return median_age_by_group.loc[row['Sex'], row['Pclass']]
  else:
    return row['Age']

df['Age'] = df.apply(fill_age, axis=1)

num_null_age = df['Age'].isnull().sum()
print(f"Number of null values in Age: {num_null_age}")

#Apply function to fill missing values

#Example bins for Age
df['Age_Binned'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, np.inf], labels=['Child', 'Teenager', 'Adult', 'Middle_Aged', 'Senior'])

#Example bins for Fare
df['Fare_Binned'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High']) #Quartile-based binning

df.head()

# Compute survival rates for Age and Fare bins
age_fare_bins_survival = df.groupby(['Age_Binned', 'Fare_Binned'], observed=False)['Survived'].mean().unstack()
print(age_fare_bins_survival)

# Find age group with highest survival rate for each fare category
max_survival_age_group_bins = age_fare_bins_survival.idxmax()
print("\nAge group with the highest survival rate for each Fare category:\n")
print(max_survival_age_group_bins)

# Plot Survival Rate Trends
plt.figure(figsize=(12, 6))

#First: Plot multiple survical rate lines for each Fare bin

#loop through each fare bin category
for fare_bin in age_fare_bins_survival.columns:
  #plot a line showing survical rates for each age bin within the given Fare bin
  plt.plot(age_fare_bins_survival.index, #x-axis
           age_fare_bins_survival[fare_bin], #y-axis
           label=f'Fare: {fare_bin}') #label for legend

for fare_bin in age_fare_bins_survival.columns:
  max_age_group = max_survival_age_group_bins[fare_bin]

  max_survival_rate = age_fare_bins_survival.loc[max_age_group, fare_bin]

  plt.scatter(max_age_group, max_survival_rate, color='purple', s=100, label=f'Max: {max_age_group} ({fare_bin}))')

#third: add labels to plot

plt.title('Survival rate by age group accross fare categories')
plt.xlabel('Age group')
plt.ylabel('surval rate')
plt.legend(title='Fare Bins', loc="upper right", fontsize='x-small')
plt.grid(True)
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# Get number if children in each dare bin
age_fare_counts = pd.crosstab(df['Age_Binned'], df['Fare_Binned'])
print(age_fare_counts.loc['Child'])

# Get edges of  fare bins
fare_bins, bin_edges = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', "High", 'Very_High'], retbins=True)

# Another way to assign binned values to df
# df['Fare_Binned'] = fare_bins

print(bin_edges)


