
########## Run parameters for all modules ##########

# HMD Credentials & data parameters 
email = "frank.oliver@live.com.ar"
password = ".kN*3BgP-!gsZ56"
country = "ITA"

#mxRates = r"C:\Users\franco.olivero\Downloads\pythonUtilities\Stoch-Mort-With-ML\Docs\ITA\STATS\Mx_1x1.txt"
mxRates = r"C:\Users\frank\Downloads\PyUtilities\Stoch-Mort-With-ML\Docs\00_ITA\STATS\Mx_1x1.txt"
genders = ["Male", "Female"]
gDict = {"Male":0, "Female":1}
gDictInv = {0:"Male", 1:"Female"}

headers = ["Year", "Age"]

minTrainYr = 1915
maxTrainYr = 2010

minTestYr = 1915
maxTestYr = 2010

minOOByr = 2011
maxOOByr = 2019

tunningFlag = True

maxAge = 100

yearsToForecast= 30

summaryFile = r"C:\Users\frank\Downloads\LC_Summary_Outputs_" + f"T{minTrainYr}-{maxTrainYr}-F{minOOByr}-{maxOOByr}.xlsx"