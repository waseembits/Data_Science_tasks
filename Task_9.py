import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic.csv")

print("Shape of dataset:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])

print("\nSummary Statistics:\n", df.describe())

df.hist(figsize=(12, 8), bins=15, color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplots of Age and Fare")
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df, palette='Set2')
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=df, palette='coolwarm')
plt.title("Survival by Gender")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Blues')
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived', palette='husl')
plt.show()

print("\nKey Insights:")
print("- Most passengers were in 3rd class.")
print("- Women had a much higher survival rate than men.")
print("- Younger passengers tended to survive more often.")
print("- Higher fare passengers were more likely to survive.")
print("- Pclass is strongly negatively correlated with survival (lower class number → higher survival).")