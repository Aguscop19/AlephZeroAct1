#Datos
#Import librerias
import kagglehub
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

if __name__ == "__main__":
    #Download dataset
    path = kagglehub.dataset_download("fedesoriano/company-bankruptcy-prediction")
    data = pd.read_csv(path+"/data.csv")

    # Standardizing the data
    scaler = StandardScaler()
    data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])


    # Saving the cleaned data
    data.to_csv("data_processing\company_bankruptcy_clean.csv", index=False)
    
    # Saving the scaler as a pickle file
    with open("data_processing/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
