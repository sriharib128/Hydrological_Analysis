from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
from discharge_analysis import perform_discharge_analysis
import ml_module  # Import the ml_module
# import dqt_module
from dqt_module import process_reservoir_data

app = Flask(__name__)
CORS(app)

def encode_plot_to_base64():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return f"data:image/png;base64,{base64.b64encode(image_png).decode()}"

def dummy_empirical_analysis(files):
    # Extract file paths
    runoff_tif_path = files.get('runoff_tif_path')
    rainfall_tif_path = files.get('rainfall_tif_path')

    # Perform the analysis
    results = perform_discharge_analysis(runoff_tif_path, rainfall_tif_path)

    return {
        'plots': results['plots'],
        'values': {
            'total_discharge': f"{results['total_discharge']:.2f} cubic feet per second",
            'reservoir_area': f"{results['reservoir_area']:.2f} sq. meters"
        }
    }

def dummy_ml_analysis(files):
    # Get the uploaded files
    train_file  = files.get('train_data')
    test_file = files.get('test_data')

    if not train_file or not test_file:
        return jsonify({'error': 'Please provide both train_data and test_data files'}), 400

    try:
        # Read the CSV files into pandas DataFrames
        train_data_df = pd.read_csv(train_file, parse_dates=['Date'])
        test_data_df = pd.read_csv(test_file, parse_dates=['Date'])
    except Exception as e:
        return jsonify({'error': f'Error reading CSV files: {str(e)}'}), 400

    # Run the analysis
    results = ml_module.run_analysis(train_data_df, test_data_df)

    # Return the results as JSON
    return results
    

def dummy_dqt_analysis(files):
    data_file  = files.get('file')
    results = process_reservoir_data(data_file, inflow_unit='m3/day', desired_T=5)
    return results

@app.route('/empirical', methods=['POST'])
def empirical():
    files = request.files
    return jsonify(dummy_empirical_analysis(files))

@app.route('/ml', methods=['POST'])
def ml():
    files = request.files
    return jsonify(dummy_ml_analysis(files))

@app.route('/dqt', methods=['POST'])
def dqt_module():
    files = request.files
    return jsonify(dummy_dqt_analysis(files))

if __name__ == '__main__':
    app.run(debug=True)