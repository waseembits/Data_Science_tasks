import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')
print(iris.head())

print(iris.info())

print(iris.describe())

print(iris.isnull().sum()) 
# no missing values 

# visulaizations 
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=iris)
plt.title('Sepal Length vs Petal Length')
plt.show()

plt.hist(iris['petal_length'], bins=20, edgecolor='black')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.show()

sns.boxplot(x='species', y='sepal_width', data=iris)
plt.title('Sepal Width by Species')
plt.show()

corr_matrix = iris.corr(numeric_only=True)
print(corr_matrix)