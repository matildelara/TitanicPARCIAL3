from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelo_titanic2.pkl')
app.logger.debug("Modelo cargado correctamente.")

@app.route('/')
def home():
    return render_template('formulario.html')  # asegúrate de tener este HTML en la carpeta 'templates'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        sex = request.form['sex']
        pclass = int(request.form['pclass'])
        embarked = request.form['embarked']

        # Crear DataFrame con los datos
        input_df = pd.DataFrame([{
            'Age': age,
            'Fare': fare,
            'Sex': sex,
            'Pclass': pclass,
            'Embarked': embarked
        }])
        app.logger.debug(f"Input recibido: \n{input_df}")

        # Hacer predicción
        pred = model.predict(input_df)[0]
        app.logger.debug(f"Resultado de predicción: {pred}")

        # Retornar resultado
        return jsonify({'sobrevive': int(pred)})
    except Exception as e:
        app.logger.error(f"Error en la predicción: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)