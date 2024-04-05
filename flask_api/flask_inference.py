from pathlib import Path
import numpy as np
import os, sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from classifiers.mlp_classifier import MLPClassifier, LEARNING_RATE, NB_EPOCHS, HIDDEN_LAYER_SIZE
from classifiers.core_functions import extract_preprocessed_datas, SEED


app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    '''Predict the label and the probability value for the input data'''
    data = request.get_json()
    data = preprocess_input(data)

    # Test the model
    prediction = pipeline.predict(data)[0]
    proba = pipeline.predict_proba(data)[0][1]
    print(prediction, proba)
    result = "promising" if prediction == 1.0 else "not promising"

    return jsonify({'prediction': result, 'proba': round(float(proba), 3)})


def preprocess_input(data):
    '''Preprocess the input data to be used by the model
    Args:
        data (dict): input data
        Returns:
        data_array (np.array): preprocessed data
    '''
    data_values = list(data.values())
    data_array = np.array(data_values).reshape(1, -1)
    return data_array


if __name__ == '__main__':
    # Load and preprocess the data
    datas_path = ".\\data\\nba_logreg.csv"
    df_features, labels = extract_preprocessed_datas(datas_path, 'TARGET_5Yrs', ['Name'])
    X_train, X_test, y_train, y_test = train_test_split(df_features.values, labels.values, test_size=0.33, random_state=SEED)
    
    current_path = Path.cwd()
    model_path = current_path / "model"

    # Load the model
    best_mlp = MLPClassifier(LEARNING_RATE, NB_EPOCHS, model_path, hidden_layer_size=HIDDEN_LAYER_SIZE)
    best_mlp.load()
    pipeline = Pipeline([
        ('scaler', MinMaxScaler().fit(X_train)),
        ('best_mlp_classifier', best_mlp)
    ])

    # Run the API
    app.run(debug=True)
