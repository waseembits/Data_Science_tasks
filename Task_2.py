import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('titanic.csv') 
print(df.head())

print(df.isnull().sum())

df['Age'].fillna(df['Age'].median(), inplace=True)


df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.drop(columns='Cabin', inplace=True)

print(f'Duplicates: {df.duplicated().sum()}')
df.drop_duplicates(inplace=True)

df['Pclass'] = df['Pclass'].astype(str)

print(df.describe())
print(df['Survived'].value_counts())

plt.hist(df['Age'], bins=30, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Boxplot: Fare
sns.boxplot(y='Fare', data=df)
plt.title('Fare Distribution')
plt.show()

sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.show()

sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Age vs Fare (Colored by Survival)')
plt.show()

sns.pairplot(df[['Survived', 'Age', 'Fare']], hue='Survived')
plt.show()

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()