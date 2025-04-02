import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
  global model
  model_path = Model.get_model_path('reglog.pkl')
  model = joblib.load(model_path)

def sigmoid(x):
  return [1 / (1 + np.exp(-y)) for y in x]

def run(raw_data):
  try: ## Try la predicción.
    test_data = pd.read_csv(raw_data, index_col=0)
    X_test = test_data.drop(columns=["Bankrupt?"])

    result = model.predict(X_test).tolist()
    result_sigmoid = sigmoid(result)
    umbral = {umbral}
    result_finals = [1 if x > umbral else 0 for x in result_sigmoid]

    return json.dumps(result_finals)
  except Exception as e:
    return json.dumps(str(e))