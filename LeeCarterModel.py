import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from scipy.linalg import svd

########## Inputs ##########

qxRatesPath = r"C:\Users\frank\Downloads\PyUtilities\Stoch-Mort-With-ML\Docs\ITA\STATS\Mx_1x1.txt" #Mortality matrix from HMD
initCalendarYear = 1850
maxAge = 110
targetFields = {"Male", "Female"}

########## 0. Preparing Data ##########

qxRates = pd.read_csv(qxRatesPath, sep="\s+", header=1)

# 0.1 Cleaning up and defining formats, setting zero to NaN
qxRates["Age"] = qxRates["Age"].replace("110+", 110).astype(int)

qxRates[["Female", "Male", "Total"]] = (
    qxRates[["Female", "Male", "Total"]]
    .astype(str)
    .replace(r"[^\d.]+", "", regex=True)
    .apply(pd.to_numeric, errors="coerce")
    .replace(0.0, np.nan)
)   

# 0.2 Filtering relevant years and target fields (Male, Female and Total)
qxRates = qxRates[qxRates["Year"]>=initCalendarYear] 
qxRates = qxRates[qxRates["Age"]<= maxAge]

########## 1. LC parameter estimation usign SVD ##########
alphaAgg = []
betaAgg = []
kappaAgg = []
agesAgg = []
gendersAgg = []
yearsAgg = []
kappaGendersAgg = []

for field in targetFields:
    # 1.1 Preparing qx matrix for SVD process 
    qxMatrix = qxRates.pivot_table(values=field, index="Age", columns="Year")

    # 1.2 Cleaning up qx matrix. For NaN, this function repeats the last value. 
    qxMatrix = qxMatrix.interpolate(axis=0, method="linear")    #Axis=0 stands for rows *it may impact the final values.
    
    """#Testing
    qxRatesPivot.to_clipboard()
    #"""

    # 1.3 Log-transform mortality rates
    qxLog = np.log(qxMatrix)
    qxLogCentered = qxLog - qxLog.mean(axis=1).values.reshape(-1,1)    #Axis=1 stands for average of all columns by row.

    # 1.4 Singular Value Decomposition (SVD) for Lee-Carter decomposition
    U, S, Vt = svd(qxLogCentered, full_matrices=False)

    """#Testing
    print("U, S, Vt Shapes: ", U.shape, S.shape, Vt.shape)
    qxLogCenteredReplica = U @ np.diag(S) @ Vt
    testSVD = np.allclose(qxLogCentered, qxLogCenteredReplica)    #Check if arrays are equal
    if testSVD: print("SVD Test Succesful for: " + field)
    #"""

    # 1.5 Extract and aggregate Lee-Carter components
    alpha_x = qxLog.mean(axis=1).values         #Average mortality across time
    beta_x = U[:, 0]/sum(U[:, 0])               #Age effect, Beta is normalized to get the unique model solution, it does not impact forecasted results though.
    kappa_t = sum(U[:, 0]) * S[0] * Vt[0, :]    #Time-varying component, adjusted by Beta normalization factor.

    alphaAgg.extend(alpha_x)
    betaAgg.extend(beta_x)
    kappaAgg.extend(kappa_t)

    gendersAgg.extend([field]*len(alpha_x))
    agesAgg.extend(qxMatrix.index.values)
    yearsAgg.extend(qxMatrix.columns.values)
    kappaGendersAgg.extend([field]*len(kappa_t))

    """#Testing
    print(type(alpha_x), alpha_x, alpha_x.shape)
    print(type(beta_x), beta_x, beta_x.shape)
    print(type(kappa_t), kappa_t, kappa_t.shape)
    #"""

########## 2. Preparing summary of LC model parameters ##########
yearsPlot = qxMatrix.columns.tolist()
agesPlot = qxMatrix.index.to_list()

aDf = pd.DataFrame({"Age":agesAgg, "Gender":gendersAgg, "Alpha":alphaAgg})
bDf = pd.DataFrame({"Age":agesAgg, "Gender":gendersAgg, "Beta":betaAgg})
kDf = pd.DataFrame({"Year":yearsAgg, "Gender": kappaGendersAgg, "Kappa":kappaAgg})

"""#Testing
alphaDf.to_clipboard()
betaDf.to_clipboard()
kappaDf.to_clipboard()
#"""

"""#Plot components
sns.relplot(x="Age", y="Alpha", data=alphaDf)
sns.relplot(x="Age", y="Beta", data=betaDf)
sns.relplot(x="Year", y="Kappa", data=kappaDf)
plt.show()
#"""
