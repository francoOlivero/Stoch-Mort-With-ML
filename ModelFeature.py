import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import seaborn as sns
import matplotlib.pyplot as plt

import LeeCarterModel
import LC_ARIMA


########## Inputs ##########

mxBEDf = LeeCarterModel.mxBEDf
mxLCDf = LC_ARIMA.mxLCFittedDf

########## Setting up ML feature ##########

mxMLDf = mxBEDf.merge(mxLCDf, on= ["Age", "Year", "Gender"], how="inner").reset_index()
mxMLDf["Cohort"] = mxMLDf["Year"] - mxMLDf["Age"]
mxMLDf["mxY"] = mxMLDf["mxLC"]/mxMLDf["mxBE"]
mxMLDf.drop(["mxLC", "mxBE"], axis=1, inplace=True)
mxMLDf['Gender'] = mxMLDf['Gender'].map({'Male': 0, 'Female': 1})

########## Setting up training and testing data ##########

def FilterByYear(df, max_year, compare):
    if compare == "<=": 
        df = df[df['Year'] <= max_year]
    else: 
        df = df[df['Year'] > max_year]
    return df

X_train = FilterByYear(mxMLDf, 2015, compare="<=")[["Year", "Age", "Cohort", "Gender"]]
y_train = FilterByYear(mxMLDf, 2015, compare="<=")[["mxY"]]

X_test = FilterByYear(mxMLDf, 2015, compare=">")[["Year", "Age", "Cohort", "Gender"]]
y_test = FilterByYear(mxMLDf, 2015, compare=">")[["mxY"]]

########## Decission Tree model ##########

dtLC = DecisionTreeRegressor(max_depth=4,  
                           min_samples_leaf=0.1, 
                           random_state=3) 

dtLC.fit(X_train, y_train)  
y_pred = dtLC.predict(X_test)  
y_pred2 = dtLC.predict(X_train)

print(y_pred2)



# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")


# Flatten y_test to Series if it's a DataFrame
y_test_flat = y_test.values.flatten()

sns.scatterplot(x=y_test_flat, y=y_pred)
plt.xlabel("Actual mxY")
plt.ylabel("Predicted mxY")
plt.title("Actual vs Predicted mxY")
plt.plot([min(y_test_flat), max(y_test_flat)], [min(y_test_flat), max(y_test_flat)], 'r--')  # identity line
plt.grid(True)
plt.show()

residuals = y_test_flat - y_pred

sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted mxY")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

sns.histplot(residuals, kde=True, bins=30)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.grid(True)
plt.show()
X_test_with_pred = X_test.copy()
X_test_with_pred['Predicted'] = y_pred
X_test_with_pred['Actual'] = y_test_flat

sns.lineplot(data=X_test_with_pred, x='Age', y='Predicted', label='Predicted')
sns.lineplot(data=X_test_with_pred, x='Age', y='Actual', label='Actual')
plt.title("mxY by Age (Predicted vs Actual)")
plt.grid(True)
plt.legend()
plt.show()
