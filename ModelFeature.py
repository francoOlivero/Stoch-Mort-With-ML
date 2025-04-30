import pandas as pd
import LeeCarterModel
import LC_ARIMA

mxBEDf = LeeCarterModel.mxBEDf
mxLCDf = LC_ARIMA.mxLCFittedDf

mxMLDf = mxBEDf.merge(mxLCDf, on= ["Age", "Year", "Gender"], how="inner").reset_index()
mxMLDf["Cohort"] = mxMLDf["Year"] - mxMLDf["Age"]
