import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , MinMaxScaler , StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
from scipy import stats

df = pd.read_csv("titanic.csv")
print(df.head())

df['Age'].fillna(df['Age'].median(), inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.drop('Cabin', axis=1, inplace=True)

df.drop_duplicates(inplace=True)

df['Pclass'] = df['Pclass'].astype(str)

df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

df['FamilySize'] = df['SibSp'] + df['Parch']

# Title from name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Simplify rare titles
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].astype('category')

# One-hot encode title
df = pd.get_dummies(df, columns=['Title'], drop_first=True)

# Drop unused columns
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)


sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot")
plt.show()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

# Drop ticket if it still exists
if 'Ticket' in df.columns:
    df.drop('Ticket', axis=1, inplace=True)

# Also drop Name / PassengerId if they are still present
for col in ['Name', 'PassengerId']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# THEN do correlation
corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot")
plt.show()