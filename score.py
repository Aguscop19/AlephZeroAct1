import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
  global model
  model_path = Model.get_model_path('model')
  model = joblib.load(model_path)

def run(raw_data):
  try: ## Try la predicción.
    # Get the data from the json 
    input_data = json.loads(raw_data)["data"][0]
    # Convert the data to a DataFrame
    input_data = pd.DataFrame(input_data)
    # Make predictions

    # This is a logistic regression model, of sklearn
    result = model.predict(input_data)
    umbral = 0.5
    result_finals = [1 if x > umbral else 0 for x in result]

    return json.dumps(result_finals)
  except Exception as e:
    return json.dumps(str(e))
