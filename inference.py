import joblib
import json
import numpy as np
import os

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "my_model.pkl"))
    return model

def input_fn(request_body, content_type):
    if content_type == "application/json":
        return np.array(json.loads(request_body))
    raise ValueError("Unsupported content type")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return json.dumps(prediction.tolist())
