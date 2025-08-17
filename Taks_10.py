import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

df = pd.read_csv("titanic.csv")

print(df.head())
print(df.info())
print(df.describe())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])  # Drop irrelevant feature

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

categorical_features = ['Sex', 'Embarked']
numeric_features = ['Age', 'Fare', 'FamilySize']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=200))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Accuracy without PCA:", accuracy_score(y_test, y_pred))

pca_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=5)),
    ('model', LogisticRegression(max_iter=200))
])
pca_pipeline.fit(X_train, y_train)
y_pred_pca = pca_pipeline.predict(X_test)
print("Accuracy with PCA:", accuracy_score(y_test, y_pred_pca))

model = pipeline.named_steps['model']
feature_names = numeric_features + list(pipeline.named_steps['preprocessor']
                                        .named_transformers_['cat'].get_feature_names_out(categorical_features))
coefficients = model.coef_[0]
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': coefficients})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()

print("\nClassification Report without PCA:\n", classification_report(y_test, y_pred))
print("\nClassification Report with PCA:\n", classification_report(y_test, y_pred_pca))
