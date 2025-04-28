import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
import tensorflow as tf

# ---------------------------
# 1. Load and preprocess data
# ---------------------------
df = pd.read_excel("nifty_final_scaled.xlsx")
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"] <= "2024-10-29"]
df = df.dropna(subset=["Return_90d"]).reset_index(drop=True)

features = df.drop(columns=["Date", "Return_90d"]).values
target = df["Return_90d"].values.reshape(-1, 1)
dates = df["Date"].values

# ---------------------------
# 2. XGBoost Regression
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train.ravel())
xgb_preds = xgb_model.predict(X_test)

# ---------------------------
# 3. LSTM Regression (MC Dropout)
# ---------------------------
lookback = 60

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

X_lstm, y_lstm = [], []
for i in range(lookback, len(features_scaled)):
    X_lstm.append(features_scaled[i - lookback:i])
    y_lstm.append(target[i])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
X_train_lstm, X_test_lstm = X_lstm[:-int(0.2 * len(X_lstm))], X_lstm[-int(0.2 * len(X_lstm)):]
y_train_lstm, y_test_lstm = y_lstm[:-int(0.2 * len(y_lstm))], y_lstm[-int(0.2 * len(y_lstm)):]
lstm_test_dates = dates[-len(X_test_lstm):]

# MC Dropout
class MCDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

# LSTM model
inp = Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]))
x = LSTM(64, return_sequences=True)(inp)
x = MCDropout(0.3)(x)
x = LSTM(32)(x)
x = MCDropout(0.3)(x)
out = Dense(1)(x)
lstm_model = Model(inputs=inp, outputs=out)
lstm_model.compile(optimizer='adam', loss='mse')

# Train
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=1)

# MC predictions
def mc_predict(model, X, n=50):
    preds = np.array([model(X, training=True).numpy().flatten() for _ in range(n)])
    return preds.mean(axis=0), preds.std(axis=0)

lstm_preds, lstm_stds = mc_predict(lstm_model, X_test_lstm)

# ---------------------------
# 4. Combine Predictions
# ---------------------------
# Align XGB and LSTM predictions to match date-wise
offset = len(X_test_lstm)
xgb_preds_aligned = xgb_preds[-offset:]

# Weighted average
hybrid_preds = 0.5 * xgb_preds_aligned + 0.5 * lstm_preds

# ---------------------------
# 5. Evaluation
# ---------------------------
def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
    return rmse, mae, r2

print("\n--- Evaluation ---")
evaluate(y_test_lstm, xgb_preds_aligned, "XGBoost")
evaluate(y_test_lstm, lstm_preds, "LSTM")
evaluate(y_test_lstm, hybrid_preds, "Hybrid")

# ---------------------------
# 6. Save results
# ---------------------------
results_df = pd.DataFrame({
    "Date": lstm_test_dates,
    "True_Return_90d": y_test_lstm.flatten(),
    "XGBoost_Pred": xgb_preds_aligned,
    "LSTM_Pred": lstm_preds,
    "LSTM_Uncertainty": lstm_stds,
    "Hybrid_Pred": hybrid_preds
})
results_df.to_csv("hybrid_xgb_lstm_predictions.csv", index=False)
print("Saved results to 'hybrid_xgb_lstm_predictions.csv'.")

from scipy.stats import norm

# ---------------------------
# 6.5 Add probability estimates from LSTM
# ---------------------------
# These assume LSTM_Pred ~ Normal(mean, std)
results_df["Prob_Return_GreaterThan_0"] = 1 - norm.cdf(0, loc=results_df["LSTM_Pred"], scale=results_df["LSTM_Uncertainty"])
results_df["Prob_Return_LessThan_Minus10"] = norm.cdf(-20, loc=results_df["LSTM_Pred"], scale=results_df["LSTM_Uncertainty"])

# Save updated results with probabilities
results_df.to_csv("hybrid_xgb_lstm_predictions_with_probs.csv", index=False)
print("Saved updated results to 'hybrid_xgb_lstm_predictions_with_probs.csv'.")


# ---------------------------
# 7. Plots
# ---------------------------
plt.figure(figsize=(14, 6))
plt.plot(results_df["True_Return_90d"].values, label="True Return", linewidth=1.5)
plt.plot(results_df["XGBoost_Pred"].values, label="XGBoost", alpha=0.7)
plt.plot(results_df["LSTM_Pred"].values, label="LSTM", alpha=0.7)
plt.plot(results_df["Hybrid_Pred"].values, label="Hybrid", linewidth=2)
plt.fill_between(range(len(results_df)), 
                 results_df["LSTM_Pred"] - results_df["LSTM_Uncertainty"],
                 results_df["LSTM_Pred"] + results_df["LSTM_Uncertainty"],
                 alpha=0.2, label="LSTM ±1 std")
plt.title("Hybrid Model: True vs Predicted 90-Day Returns")
plt.xlabel("Test Set Index")
plt.ylabel("Return_90d")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
