# ---------------------------
# 1. Import Libraries
# ---------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# ---------------------------
# 2. Load and Preprocess Data
# ---------------------------
# Load dataset
df = pd.read_excel("nifty_final_scaled.xlsx")

# Drop NaN rows
df = df.dropna().reset_index(drop=True)

# Define features and target
features = df.drop(columns=["Date", "Return_90d"]).columns.tolist()
target = "Return_90d"

# Standardize input features (just to be extra careful, though they are already scaled)
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# ---------------------------
# 3. Create Sequences for LSTM
# ---------------------------
window_size = 30  # 30 days lookback window

X = []
y = []

for i in range(len(df) - window_size):
    X.append(df[features].iloc[i:i + window_size].values)
    y.append(df[target].iloc[i + window_size])

X = np.array(X)
y = np.array(y)

# ---------------------------
# 4. Train-Test Split (80-20 time-based)
# ---------------------------
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ---------------------------
# 5. Build the LSTM Model
# ---------------------------
model = Sequential([
    LSTM(64, input_shape=(window_size, len(features)), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# ---------------------------
# 6. Train the Model
# ---------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ---------------------------
# 7. Predict and Evaluate
# ---------------------------
y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"LSTM Test MAE: {mae:.4f}%")
print(f"LSTM Test RMSE: {rmse:.4f}%")
print(f"LSTM Test R²: {r2:.4f}")

# ---------------------------
# 8. Plot Actual vs Predicted
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual", color='blue', linewidth=2)
plt.plot(y_pred, label="Predicted (LSTM)", color='red', linestyle='--', linewidth=2)
plt.title("LSTM: Actual vs Predicted 90-Day Returns", fontsize=16)
plt.xlabel("Test Sample Index", fontsize=14)
plt.ylabel("Return (%)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
