from flask import Flask, render_template, request
import pickle
import numpy as np

# Flask app create
app = Flask(__name__)

# Load the model
model_path = "model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            x = float(request.form['x'])
            y = float(request.form['y'])
            z = float(request.form['z'])
          

            data = np.array([[x, y, z]])
            prediction = model.predict(data)[0]

            return render_template('results.html', prediction=prediction)

        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)