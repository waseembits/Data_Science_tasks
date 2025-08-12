import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

n_samples = 20640
n_features = 8
X, y = make_regression(n_samples=n_samples, n_features=n_features,
                       noise=0.8, random_state=42, bias=5.0)

feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                 'Population', 'AveOccup', 'Latitude', 'Longitude'][:n_features]

X = pd.DataFrame(X, columns=feature_names)
rng = np.random.RandomState(42)
X['MedInc'] = np.abs(X['MedInc']) * 2.5
X['HouseAge'] = np.clip((X['HouseAge'] % 50) + rng.randn(n_samples)*2, 1, 50)
X['AveRooms'] = np.abs(X['AveRooms']) * 1.5 + 1
X['AveBedrms'] = np.abs(X['AveBedrms']) * 0.8 + 0.5
X['Population'] = np.abs(X['Population']) * 50 + 10
X['AveOccup'] = np.clip(np.abs(X['AveOccup']) * 3 + 1, 0.5, 10)
X['Latitude'] = rng.uniform(32, 42, size=n_samples)
X['Longitude'] = rng.uniform(-124, -114, size=n_samples)

y = (y - y.min()) / (y.max() - y.min()) * 4 + 0.5
y = pd.Series(y, name='MedHouseVal')

df = pd.concat([X, y], axis=1)
print("First 5 rows:\n", df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Evaluation on Test Set ===")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")


coefficients = pd.Series(lr.coef_, index=feature_names).sort_values(key=abs, ascending=False)
print("\nFeature Importance (Coefficients):\n", coefficients)

residuals = y_test - y_pred

alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
ridge_best_alpha = ridge_cv.alpha_
print(f"\nBest alpha for Ridge: {ridge_best_alpha}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lr_cv_scores = cross_val_score(LinearRegression(), scaler.transform(X), y, scoring='r2', cv=kf)
ridge_cv_scores = cross_val_score(Ridge(alpha=ridge_best_alpha), scaler.transform(X), y, scoring='r2', cv=kf)
print(f"LR CV mean R²: {lr_cv_scores.mean():.4f}")
print(f"Ridge CV mean R²: {ridge_cv_scores.mean():.4f}")

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

corr = df.corr()
plt.figure(figsize=(8,6))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle='--', color='red')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()

plt.figure(figsize=(7,4))
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Residual Distribution")
plt.show()

plt.figure(figsize=(8,5))
coefficients.plot(kind='bar')
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Feature Importance")
plt.show()

with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump({"model": lr, "scaler": scaler, "features": feature_names}, f)

print("\nModel saved as 'linear_regression_model.pkl'")