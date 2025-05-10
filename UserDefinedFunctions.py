import numpy as np
import pandas as pd


from scipy.linalg import svd

def LeeCarterSVD(mxMatrix: pd.DataFrame):
    """
    Perform Lee-Carter decomposition on a matrix of mortality rates (mxMatrix).
    
    Parameters:
        mxMatrix (pd.DataFrame): Rows are ages, columns are years. Values are central death rates.

    Returns:
        ax (pd.Series): Average log mortality by age.
        bx (pd.Series): Age-specific sensitivity to mortality trends. Beta is normalized to unit. then Sum.bx = 1
        kt (pd.Series): Time-varying mortality index.
    """
    # Step 1: Take log and center the matrix
    mxLog = np.log(mxMatrix)
    mxLogCentered = mxLog.sub(mxLog.mean(axis=1), axis=0)

    # Step 2: SVD decomposition
    U, S, Vt = svd(mxLogCentered, full_matrices=False)

    # Step 3: Extract Lee-Carter components
    bRaw = U[:, 0]
    
    ax = mxLog.mean(axis=1)
    bx = bRaw / bRaw.sum()
    kt = bRaw.sum() * S[0] * Vt[0, :]
    
    # Check that SVD works well 
    assert np.allclose(mxLogCentered, U @ np.diag(S) @ Vt) 
    
    return ax, bx, kt


from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from scipy.stats import jarque_bera,skew, kurtosis

def ARIMAsTests(resid):

    ljung = acorr_ljungbox(resid, lags=[1], return_df=True)
    jb_stat, jb_pval = jarque_bera(resid)
    
    return {
        "LJ_Box_Stat": ljung['lb_stat'].iloc[0],            #Ljung-Box Q-Stat
        "LJung_Box_P-Value": ljung['lb_pvalue'].iloc[0],    #Ljung-Box P-Value
        "JB_Stat": jb_stat,                                 #Jarque-Bera Stat
        "JB_P-Value": jb_pval,                              #Jarque-Bera P-Value
        "ADF_Resid_Stat": adfuller(resid)[0],               #Augmented Dickey-Fuller Stat
        "ADF_Resid_P-Value": adfuller(resid)[1],            #Augmented Dickey-Fuller         
        "Skew": skew(resid),                                #Skewness
        "Kurtosis": kurtosis(resid),                        #Kurtosis
    }
     
def ARIMAsGrid(kARIMAs):
    """
    Summarizes a list of ARIMA models into a single DataFrame.

    Parameters:
        kARIMAs (list): List of fitted ARIMA models (e.g., from pmdarima.auto_arima with return_valid_fits=True).
    
    Returns:
        DataFrame: Combined summary of model orders, scores, diagnostics, and parameters.
    """
    kARIMARecords = []

    for model in kARIMAs:
        kARIMARecords.append({
            "ARIMA Order": model.order,
            "AIC": model.aic(),
            "BIC": model.bic(),
            "HQIC": model.hqic(),
            "MSE": model.arima_res_.mse,
            "MAE": model.arima_res_.mae,
            "params": model.params().to_dict(),
            "diags": ARIMAsTests(model.arima_res_.resid)
        })

    df = pd.concat([
        pd.DataFrame(kARIMARecords).drop(["params", "diags"], axis=1),
        pd.json_normalize([record["diags"] for record in kARIMARecords]),
        pd.json_normalize([record["params"] for record in kARIMARecords])
    ], axis=1)
  
    return df
