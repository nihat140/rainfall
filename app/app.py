from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the trained model
model = joblib.load('rf_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        year = int(request.form['year'])
        month = int(request.form['month'])
        rfh_lag1 = float(request.form['rfh_lag1'])

        # Prepare input for model
        input_data = np.array([[year, month, rfh_lag1]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]

        # Generate a simple plot (actual vs. predicted for demo)
        actual_data = pd.read_csv('ken-rainfall-adm2-full.csv').iloc[1:]
        actual_data['date'] = pd.to_datetime(actual_data['date'])
        actual_data['rfh'] = pd.to_numeric(actual_data['rfh'], errors='coerce')
        recent_data = actual_data.tail(12)[['date', 'rfh']].dropna()

        plt.figure(figsize=(10, 5))
        plt.plot(recent_data['date'], recent_data['rfh'], label='Actual', marker='o')
        plt.plot(pd.to_datetime(f'{year}-{month}-01'), prediction, 'r*', label='Predicted', markersize=15)
        plt.title('Rainfall Prediction')
        plt.xlabel('Date')
        plt.ylabel('Rainfall (mm)')
        plt.legend()
        plt.grid(True)

        # Save plot to a bytes buffer
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        return jsonify({
            'prediction': round(prediction, 2),
            'plot_url': plot_url,
            'mae': 10.5  # Replace with your actual MAE from training
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)