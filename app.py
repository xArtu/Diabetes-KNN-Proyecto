from flask import Flask, render_template, request
import pickle
import numpy as np

# Cargar el modelo entrenado
model_path = "model/knn_diabetes_model.pkl"
scaler_path = "model/scaler.pkl"  # Ruta al scaler usado durante el entrenamiento

with open(model_path, "rb") as file:
    knn_model = pickle.load(file)

with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)  # Cargar el MinMaxScaler entrenado

# Crear la app de Flask
app = Flask(__name__)

# Mapeos para transformar valores de lenguaje natural a numéricos
gender_mapping = {"Female": 0, "Male": 1}
smoking_mapping = {
    "never": 0,
    "No Info": 1,
    "former": 1,
    "current": 2,
    "not current": 2,
    "ever": 1,
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Obtener los datos del formulario
        gender = request.form.get("gender")
        age = float(request.form.get("age"))
        hypertension = int(request.form.get("hypertension"))
        heart_disease = int(request.form.get("heart_disease"))
        smoking_history = request.form.get("smoking_history")
        bmi = float(request.form.get("bmi"))
        HbA1c_level = float(request.form.get("HbA1c_level"))
        blood_glucose_level = float(request.form.get("blood_glucose_level"))

        # Transformar valores a numéricos
        gender_num = gender_mapping.get(gender, 0)
        smoking_num = smoking_mapping.get(smoking_history, 0)

        # Crear un array para la predicción
        input_data = np.array([[gender_num, age, hypertension, heart_disease, 
                                smoking_num, bmi, HbA1c_level, blood_glucose_level]])

        # Normalizar las columnas numéricas (usar el scaler entrenado)
        numerical_columns = [1, 5, 6, 7]  # Columnas: age, bmi, HbA1c_level, blood_glucose_level
        input_data[:, numerical_columns] = scaler.transform(input_data[:, numerical_columns])

        # Realizar la predicción
        prediction = knn_model.predict(input_data)
        result = "Es probable que tenga diabetes" if prediction[0] == 1 else "Es probable que no tenga diabetes"

        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=False)