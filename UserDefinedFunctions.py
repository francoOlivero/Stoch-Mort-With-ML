import pandas as pd
import numpy as np

from scipy.linalg import svd

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from scipy.stats import jarque_bera,skew, kurtosis

import requests
from bs4 import BeautifulSoup
from io import StringIO

from openpyxl import load_workbook
import os

def FilterByYr(df, max_year, compare):
    if compare == "<=": 
        df = df[df['Year'] <= max_year]
    else: 
        df = df[df['Year'] > max_year]
    return df

def LeeCarterSVD(mxMatrix: pd.DataFrame, normalizeFlag=True):
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
     
    if normalizeFlag == True:
        bx = bRaw / bRaw.sum()
        kt = bRaw.sum() * S[0] * Vt[0, :]
    else:
        bx = bRaw
        kt = S[0] * Vt[0, :]   
    
    # Check that SVD works well 
    assert np.allclose(mxLogCentered, U @ np.diag(S) @ Vt) 
    
    print("Variance Explained by first singular value: ", S[0]**2 / np.sum(S**2))
    
    return ax, bx, kt

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

def getMxFromHMD(email, password, country):
    
    # Start session
    session = requests.Session()

    # Step 1: Get verification token
    loginURL = 'https://www.mortality.org/Account/Login'
    loginPage = session.get(loginURL)
    soup = BeautifulSoup(loginPage.content, 'html.parser')
    token = soup.find('input', {'name': '__RequestVerificationToken'})['value']

    # Step 2: Send login request
    payload = {
        'Email': email,
        'Password': password,
        '__RequestVerificationToken': token
    }
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': loginURL
    }
    response = session.post(loginURL, data=payload, headers=headers)

    # Step 3: Check login success
    if "Logout" not in response.text and "/Account/Logout" not in response.text:
        print("Login failed.")
        exit()

    print("Login successful.")

    # Step 4: Access dataset
    dataURL = f'https://www.mortality.org/File/GetDocument/hmd.v6/{country}/STATS/Mx_1x1.txt'
    dataResponse = session.get(dataURL)

    if dataResponse.status_code == 200:
        print("Dataset downloaded successfully.")
    else:
        print(f"Failed to download dataset. Status code: {dataResponse.status_code}")
        exit()

    # Step 5: Convert content to DataFrame, sep='\s+' to separate columns by spaces
    rawData = dataResponse.text
    df = pd.read_csv(StringIO(rawData),  sep='\s+', skiprows=1) 
    
    return df

def add_transformed_cols(df, targetCols, function, prefix=''):
    """
    Applies a transformation function to specified columns in a DataFrame.

    Parameters:
    - df: pd.DataFrame
    - targetCols: list of column names to transform
    - function: function to apply to the columns (e.g., np.log)
    - prefix: prefix to add to new column names

    Returns:
    - df: original DataFrame with new transformed columns added
    """
    for col in targetCols:
        newCol = f"{prefix}{col}"
        df[newCol] = function(df[col])
    return df

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan
    y_true_c = np.clip(y_true[mask], eps, None)
    rel_err = np.abs((y_true_c - y_pred[mask]) / y_true_c)
    return float(np.mean(rel_err) * 100.0)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))

def rmsle_from_logs(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    # RMSLE como RMSE en espacio log (usando columnas log_ provistas)
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    if not np.any(mask):
        return np.nan
    diff = y_true_log[mask] - y_pred_log[mask]
    return float(np.sqrt(np.mean(diff**2)))

def save_df_to_excel(excel_path: str, df: pd.DataFrame, sheet_name: str):
    """
    Guarda un DataFrame en una hoja específica de un archivo Excel (.xlsx).
    Si la hoja ya existe, la reemplaza.
    Compatible con pandas >= 2.0
    """

    try:
        # Si el archivo existe -> agregar (reemplazando hoja si existe)
        if os.path.exists(excel_path):
            with pd.ExcelWriter(
                excel_path,
                engine="openpyxl",
                mode="a",  # append
                if_sheet_exists="replace"  # reemplaza la hoja si ya existe
            ) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"✅ Hoja '{sheet_name}' actualizada correctamente en '{excel_path}'.")

        else:
            # Si el archivo no existe -> crear nuevo
            df.to_excel(excel_path, sheet_name=sheet_name, index=False)
            print(f"📄 Archivo nuevo creado con hoja '{sheet_name}' en '{excel_path}'.")

    except Exception as e:
        print(f"❌ Error al guardar el DataFrame: {e}")
