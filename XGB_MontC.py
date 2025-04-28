import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy.stats import norm
import matplotlib.pyplot as plt

# -----------------------------------
# Step 1: Load and preprocess dataset
# -----------------------------------
df = pd.read_excel("nifty_final_scaled.xlsx")
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"] <= "2024-10-29"]
df = df.dropna(subset=["Return_90d"]).reset_index(drop=True)

# Extract features and target
X = df.drop(columns=["Date", "Return_90d"]).values
y = df["Return_90d"].values
dates = df["Date"].values

# Time-based split (no shuffle)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
test_dates = dates[split_idx:]

# --------------------------------------------
# Step 2: Monte Carlo via Bootstrap + XGBoost
# --------------------------------------------
n_models = 50  # Number of MC simulations
predictions = []

for i in range(n_models):
    X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=i)
    
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=i,
        n_jobs=-1
    )
    
    model.fit(X_boot, y_boot)
    preds = model.predict(X_test)
    predictions.append(preds)

predictions = np.array(predictions)  # shape (n_models, n_samples)

# --------------------------------------------
# Step 3: Compute Monte Carlo Statistics
# --------------------------------------------
y_mean = predictions.mean(axis=0)
y_std = predictions.std(axis=0)

# Probabilities
p_up = 1 - norm.cdf(0, loc=y_mean, scale=y_std)
p_crash = norm.cdf(-20, loc=y_mean, scale=y_std)

# --------------------------------------------
# Step 4: Evaluation
# --------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_mean))
mae = mean_absolute_error(y_test, y_mean)
r2 = r2_score(y_test, y_mean)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# --------------------------------------------
# Step 5: Save Detailed Results to CSV
# --------------------------------------------
results_df = pd.DataFrame({
    "Date": test_dates,
    "True_Return_90d": y_test,
    "Predicted_Mean": y_mean,
    "Uncertainty_Std": y_std,
    "Prob_Return_Positive": p_up,
    "Prob_Return_LessThan-20": p_crash
})

results_df.to_csv("xgboost_mc_predictions_3.csv", index=False)
print("Saved results to 'xgboost_mc_predictions.csv'.")

# Optional: Print a preview
print(results_df.head())

# --------------------------------------------
# Step 6: Plot predictions with uncertainty
# --------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="True Return_90d", linewidth=1.5)
plt.plot(y_mean, label="Predicted Mean", linewidth=1.5)
plt.fill_between(range(len(y_mean)), y_mean - y_std, y_mean + y_std, alpha=0.3, label="±1 std dev")
plt.title("XGBoost Monte Carlo: Prediction + Uncertainty")
plt.xlabel("Test Set Index")
plt.ylabel("Return_90d")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
