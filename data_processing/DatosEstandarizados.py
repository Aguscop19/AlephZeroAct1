#Datos
#Import librerias
import kagglehub
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Download dataset
path = kagglehub.dataset_download("fedesoriano/company-bankruptcy-prediction")
data = pd.read_csv(path+"/data.csv")

# Exploring the dataset
print(data.info())
print(data.describe())
print(data.isnull().sum()) 

# Standardizing the data
scaler = StandardScaler()
data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])

# Saving the cleaned data
data.to_csv("company_bankruptcy_clean.csv", index=False)