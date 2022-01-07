from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import traceback
from flask_restful import reqparse
import h5py
from keras.models import load_model

app = Flask(__name__)
@app.route("/", methods=['GET'])
def hello():
    return "hey"
@app.route('/predict', methods=['POST'])
def predict():
    model = load_model('/workspace/flash_amr_bsc/experiment-1-window-300-scaler-min-max-scaler-(-1, 1)-train-acc-0.88457-val-acc-0.94108.hdf5')

    # lr = joblib.load("model.pkl")
    if model:
        try:
            json = request.get_json()  
            model_columns = ["dirt_road", "cobblestone_road", "asphalt_road"]
            print(json)
            
            temp=list(json['arr'])
            vals=np.array(temp)
            print(temp)
            vals = np.expand_dims(vals, axis=0)
            prediction = model.predict(vals)
            print("here:",prediction)        
            return jsonify({'prediction': str(prediction[0])})
        except:        
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')
if __name__ == '__main__':
    app.run(debug=True)
