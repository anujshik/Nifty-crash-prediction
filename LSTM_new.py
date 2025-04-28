import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# ----------------------------
# Load and prepare data
# ----------------------------
df = pd.read_excel("nifty_final_scaled.xlsx")
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"] <= "2024-10-29"]
df = df.dropna(subset=["Return_90d"]).reset_index(drop=True)

# Drop Date and extract features/target
features = df.drop(columns=["Date", "Return_90d"]).values
target = df["Return_90d"].values.reshape(-1, 1)

# Scale features
scaler_X = MinMaxScaler()
features_scaled = scaler_X.fit_transform(features)

# ----------------------------
# Create LSTM sequences
# ----------------------------
lookback = 90

X_lstm, y_lstm = [], []
for i in range(lookback, len(features_scaled)):
    X_lstm.append(features_scaled[i-lookback:i])
    y_lstm.append(target[i])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# ----------------------------
# Train-test split (80-20, no shuffle)
# ----------------------------
split = int(0.8 * len(X_lstm))
X_train, X_test = X_lstm[:split], X_lstm[split:]
y_train, y_test = y_lstm[:split], y_lstm[split:]

# ----------------------------
# Build LSTM model
# ----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ----------------------------
# Train the model
# ----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ----------------------------
# Predict and Evaluate
# ----------------------------
y_pred = model.predict(X_test).flatten()
y_true = y_test.flatten()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"\nRMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# ----------------------------
# Plot Results
# ----------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_true, label="True Return_90d")
plt.plot(y_pred, label="Predicted Return_90d")
plt.title("LSTM: Predicted vs True 90-Day Returns")
plt.xlabel("Test Set Index")
plt.ylabel("Return_90d")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
