
########## Run parameters for all modules ##########

# HMD Credentials & data parameters 
email = "frank.oliver@live.com.ar"
password = ".kN*3BgP-!gsZ56"
country = "ITA"

mxRates = r"C:\Users\franco.olivero\Downloads\pythonUtilities\Stoch-Mort-With-ML\Docs\ITA\STATS\Mx_1x1.txt"
#mxRates = r"C:\Users\frank\Downloads\PyUtilities\Stoch-Mort-With-ML\Docs\ITA\STATS\Mx_1x1.txt"
genders = ["Male", "Female"]
gDict = {"Male":0, "Female":1}
gDictInv = {0:"Male", 1:"Female"}

headers = ["Year", "Age"]

initCalendarYear = 1900

minTrainYr = 1900
maxTrainYr = 2015

maxAge = 100
yearsToForecast= 20