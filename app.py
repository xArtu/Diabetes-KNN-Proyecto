from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Cargar el modelo, scaler y mappings
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
gender_mapping = joblib.load('gender_mapping.pkl')
smoking_history_mapping = joblib.load('smoking_history_mapping.pkl')

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = request.form['smoking_history']
        bmi = float(request.form['bmi'])
        hba1c = float(request.form['hba1c'])
        blood_glucose = int(request.form['blood_glucose'])

        # Crear las características
        features = [
            gender_mapping[gender],
            age,
            hypertension,
            heart_disease,
            smoking_history_mapping[smoking_history],
            bmi,
            hba1c,
            blood_glucose
        ]

        # Escalar las características
        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features)

        # Realizar la predicción
        prediction = knn_model.predict(scaled_features)

        # Determinar el resultado
        if prediction[0] == 1:
            result = "Es probable que la persona tenga diabetes."
        else:
            result = "Es poco probable que la persona tenga diabetes."

        return render_template('index.html', result=result)

    except Exception as e:
        error = f"Ocurrió un error: {str(e)}"
        return render_template('index.html', error=error)

if __name__ == '__main__':
    app.run(debug=True)