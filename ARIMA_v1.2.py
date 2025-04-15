import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from scipy.linalg import svd

# Example: Synthetic mortality data (replace with your actual dataset)
# Rows represent ages, and columns represent years

np.random.seed(42)
ages = np.arange(50, 101)  # Ages 50 to 100
years = np.arange(2000, 2021)  # Years 2000 to 2020
qx_rates = np.exp(-0.02 * (ages[:, None] - 50) + 0.005 * (years - 2000) + np.random.normal(0, 0.02, (len(ages), len(years))))
qx_df = pd.DataFrame(qx_rates, index=ages, columns=years)

# Plot the mortality surface
plt.figure(figsize=(12, 6))
plt.imshow(qx_rates, aspect='auto', cmap='viridis', extent=[years[0], years[-1], ages[-1], ages[0]])
plt.colorbar(label='Mortality Rate')
plt.title("Mortality Rates by Age and Year")
plt.xlabel("Year")
plt.ylabel("Age")
#plt.show()

# Step 1: Log-transform mortality rates
log_qx = np.log(qx_df)

log_qx_centered = log_qx - log_qx.mean(axis=1).values[:, None]

# Step 2: Singular Value Decomposition (SVD) for Lee-Carter decomposition
U, S, Vt = svd(log_qx_centered)

# Extract Lee-Carter components
alpha_x = log_qx_centered.mean(axis=1)  # Average mortality across time
beta_x = U[:, 0]  # Age effect
kappa_t = S[0] * Vt[0, :]  # Time-varying component


print(type(beta_x), beta_x)
print(type(kappa_t), kappa_t)
print(type(alpha_x), alpha_x)

# Plot components
""" plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(ages, alpha_x, label="Alpha (α)", color="blue")
plt.title("Alpha (Average Log Mortality)")
plt.xlabel("Age")
plt.ylabel("Value")

plt.subplot(1, 3, 2)
plt.plot(ages, beta_x, label="Beta (β)", color="orange")
plt.title("Beta (Age Effect)")
plt.xlabel("Age")
plt.ylabel("Value")

plt.subplot(1, 3, 3)
plt.plot(years, kappa_t, label="Kappa (κ)", color="green")
plt.title("Kappa (Time Effect)")
plt.xlabel("Year")
plt.ylabel("Value")

plt.tight_layout()
plt.show() """

# Step 3: Fit ARIMA model to kappa_t (time-varying component)
p, d, q = 1, 1, 1  # Example ARIMA parameters
LC_ARIMA = ARIMA(kappa_t, order=(p, d, q))
LC_ARIMA_fitted = LC_ARIMA.fit()

print(LC_ARIMA_fitted.summary())

# Forecast future kappa_t
forecast_steps = 10
forecast_kappa = fitted_model.get_forecast(steps=forecast_steps)
forecast_index = np.arange(years[-1] + 1, years[-1] + 1 + forecast_steps)
forecast_mean = forecast_kappa.predicted_mean
forecast_conf_int = forecast_kappa.conf_int()

# Plot kappa_t and forecast
plt.figure(figsize=(10, 6))
plt.plot(years, kappa_t, label="Observed Kappa (κ)", color="green")
plt.plot(forecast_index, forecast_mean, label="Forecast Kappa (κ)", color="orange")
plt.fill_between(
    forecast_index,
    forecast_conf_int.iloc[:, 0],
    forecast_conf_int.iloc[:, 1],
    color="orange",
    alpha=0.2,
    label="Confidence Interval",
)
plt.title("Forecast of Kappa (Time Effect)")
plt.xlabel("Year")
plt.ylabel("Kappa (κ)")
plt.legend()
plt.show()

# Step 4: Reconstruct mortality rates for forecast years
forecast_log_mortality = alpha_x[:, None] + beta_x[:, None] @ forecast_mean[None, :]
forecast_mortality = np.exp(forecast_log_mortality)

# Combine historical and forecasted mortality rates
all_years = np.concatenate([years, forecast_index])
all_mortality = np.hstack([qx_df.values, forecast_mortality])

# Plot historical and forecasted mortality rates
plt.figure(figsize=(12, 6))
plt.imshow(all_mortality, aspect='auto', cmap='viridis', extent=[all_years[0], all_years[-1], ages[-1], ages[0]])
plt.colorbar(label='Mortality Rate')
plt.title("Historical and Forecasted Mortality Rates")
plt.xlabel("Year")
plt.ylabel("Age")
plt.show()
