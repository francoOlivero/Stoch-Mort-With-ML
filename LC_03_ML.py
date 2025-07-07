import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import seaborn as sns
import matplotlib.pyplot as plt

import RunParameters as rp
import UserDefinedFunctions as udf
import LC_01_BaseModel as lc1
import LC_02_kARIMA as lc2

########## 0. Inputs ##########
gDict = rp.gDict
gDictInv = rp.gDictInv
mxBEDf = lc1.mxBEDf
mxLC_Base_Df = lc2.mxLC_Base_Df

########## Setting up ML feature ##########
mx_X = mxBEDf.merge(mxLC_Base_Df, on= ["Age", "Year", "Gender"], how="inner").reset_index()
mx_X["Cohort"] = mx_X["Year"] - mx_X["Age"]
mx_X["mx_Y"] = mx_X["mx_LC"]/mx_X["mx_BE"]
mx_X.drop(["mx_LC", "mx_BE"], axis=1, inplace=True)
mx_X["Gender"] = mx_X["Gender"].map(gDict)

########## Defining Training and Testing data ##########
X_train = udf.FilterByYear(mx_X, 2015, compare="<=")[["Year", "Age", "Cohort", "Gender"]]
y_train = udf.FilterByYear(mx_X, 2015, compare="<=")[["mx_Y"]]

X_test = udf.FilterByYear(mx_X, 2015, compare=">")[["Year", "Age", "Cohort", "Gender"]]
y_test = udf.FilterByYear(mx_X, 2015, compare=">")[["mx_Y"]]

########## Decission Tree model ##########
mY_DT = DecisionTreeRegressor(max_depth=4,  
                           min_samples_leaf=0.1, 
                           random_state=3) 

mY_DT.fit(X_train, y_train)  
y_pred = mY_DT.predict(X_test)
y_pred_train = mY_DT.predict(X_train)

mY_DT_Df = X_train
mY_DT_Df["mx_Y_DT"] = y_pred_train
mY_DT_Df["mx_Y"] = y_train
mY_DT_Df["Gender"] = mY_DT_Df["Gender"].map(gDictInv)


"""#Testing 
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
plt.xlabel("Actual mx_Y")
plt.ylabel("Predicted mx_Y")
plt.title("Actual vs Predicted mx_Y")
plt.plot([min(y_test_flat), max(y_test_flat)], [min(y_test_flat), max(y_test_flat)], 'r--')  # identity line
plt.grid(True)
plt.show()

residuals = y_test_flat - y_pred

sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted mx_Y")
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
plt.title("mx_Y by Age (Predicted vs Actual)")
plt.grid(True)
plt.legend()
plt.show()
#"""