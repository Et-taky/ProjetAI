from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load models and scaler
log_reg = joblib.load('models/log_reg.pkl')
knn = joblib.load('models/knn.pkl')
tree = joblib.load('models/tree.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

models = {
    'Logistic Regression': log_reg,
    'KNN': knn,
    'Decision Tree': tree
}

@app.route('/')
def index():
    return render_template('index.html', features=feature_names, models=models.keys())

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = [float(request.form[feature]) for feature in feature_names]
    model_name = request.form['model']
    
    # Scale input
    scaled_input = scaler.transform([input_data])
    
    # Predict
    model = models[model_name]
    prediction = model.predict(scaled_input)[0]

    result = "Malignant" if prediction == 1 else "Benign"
    
    return render_template('result.html', result=result, model=model_name)

if __name__ == '__main__':
    app.run(debug=True)
