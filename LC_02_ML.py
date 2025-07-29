import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import seaborn as sns
import matplotlib.pyplot as plt

import RunParameters as rp
import UserDefinedFunctions as udf
import LC_01_BaseModel as lc1

########## 0. Inputs ##########
gDict = rp.gDict
gDictInv = rp.gDictInv
maxTrainYr = rp.maxTrainYr
mxBEDf = lc1.mxBEDf
mxLC_Base_Df = lc1.mxLC_Base_Df

########## 1. Setting up ML feature ##########
mx_X = mxBEDf.merge(mxLC_Base_Df, on= ["Age", "Year", "Gender"], how="inner").reset_index()
mx_X["Gender"] = mx_X["Gender"].map(gDict)
mx_X.insert(loc=3, column='Cohort', value= mx_X["Year"] - mx_X["Age"])
mx_X["mx_Y_LC"] = mx_X["mx_BE"]/mx_X["mx_LC"]

########## 2.Defining Training and Testing data ##########

X_train = udf.FilterByYr(mx_X, maxTrainYr, compare="<=")[["Year", "Age", "Cohort", "Gender"]] #DF
y_train = udf.FilterByYr(mx_X, maxTrainYr, compare="<=")["mx_Y_LC"] #Series

X_test = udf.FilterByYr(mx_X, maxTrainYr, compare=">")[["Year", "Age", "Cohort", "Gender"]] #DF
y_test = udf.FilterByYr(mx_X, maxTrainYr, compare=">")["mx_Y_LC"] #Series

########## 3.ML models and metrics for analysis ##########
mY_DT = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=0.10,
    random_state=3
    )
 
mY_RF = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=0.10,
    random_state=3
    )

mY_GB = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=4,
    random_state=3
    )

# 3.1 Training models
mY_DT.fit(X_train, y_train)
mY_RF.fit(X_train, y_train)  
mY_GB.fit(X_train, y_train)    

# 3.2 Summary df with ML outputs
mY_ML_Df = udf.FilterByYr(mx_X, maxTrainYr, compare="<=").copy()
mY_ML_Df["Gender"] = mY_ML_Df["Gender"].map(gDictInv)
mY_ML_Df["mx_Y_DT"] = mY_DT.predict(X_train)
mY_ML_Df["mx_Y_RF"] = mY_RF.predict(X_train)
mY_ML_Df["mx_Y_GB"] = mY_GB.predict(X_train)

# 3.3 adding deltas to measure mort adjustments  
mY_ML_Df = udf.add_transformed_cols(
    df= mY_ML_Df,
    targetCols=["mx_Y_LC", "mx_Y_DT", "mx_Y_RF", "mx_Y_GB"], 
    function= lambda x: x-1, 
    prefix="delta_"
    )

# 3.4 Calculating LeeCarter-ML mx´s 
mY_ML_Df = udf.add_transformed_cols(
    df= mY_ML_Df,
    targetCols=["mx_Y_DT", "mx_Y_RF", "mx_Y_GB"], 
    function= lambda x: x * mY_ML_Df["mx_LC"], 
    prefix="lc_"
    )

# 3.5 Calculating LeeCarter-ML log mx´s 
mY_ML_Df = udf.add_transformed_cols(
    df= mY_ML_Df,
    targetCols=["mx_BE", "mx_LC", "lc_mx_Y_DT", "lc_mx_Y_RF", "lc_mx_Y_GB"], 
    function= np.log, 
    prefix="log_"
    )

mY_ML_Df.to_clipboard()

########## 4.Analyzing ML results ##########
deltaPlot = sns.heatmap(
                        mY_ML_Df[mY_ML_Df["Gender"]=="Male"].pivot(index="Year",columns="Age", values="delta_mx_Y_DT"), 
                        fmt="d",
                        )

plt.show()

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