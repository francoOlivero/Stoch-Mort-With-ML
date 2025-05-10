import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

import RunParameters as rp


email = rp.email
password = rp.password

# Start session
session = requests.Session()

# Step 1: Get verification token
login_url = 'https://www.mortality.org/Account/Login'
login_page = session.get(login_url)
soup = BeautifulSoup(login_page.content, 'html.parser')
token = soup.find('input', {'name': '__RequestVerificationToken'})['value']

# Step 2: Send login request
payload = {
    'Email': email,
    'Password': password,
    '__RequestVerificationToken': token
}
headers = {
    'User-Agent': 'Mozilla/5.0',
    'Referer': login_url
}
response = session.post(login_url, data=payload, headers=headers)

# Step 3: Check login success
if "Logout" not in response.text and "/Account/Logout" not in response.text:
    print("Login failed.")
    exit()

print("Login successful.")

# Step 4: Access dataset
data_url = 'https://www.mortality.org/File/GetDocument/hmd.v6/ITA/InputDB/ITAdeath.txt'
data_response = session.get(data_url)

if data_response.status_code == 200:
    print("Dataset downloaded successfully.")
else:
    print(f"Failed to download dataset. Status code: {data_response.status_code}")
    exit()

# Step 5: Convert content to DataFrame
raw_data = data_response.text
df = pd.read_csv(StringIO(raw_data), sep=r',', comment='#', engine='python')
df.to_clipboard()
print(df.head())
