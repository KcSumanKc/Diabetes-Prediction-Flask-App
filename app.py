
   # ---------
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

# Define the PyTorch model class
class ANN_Model(nn.Module):
    def __init__(self, input_features=8, hidden1=20, hidden2=20, out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_features)

    def forward(self, X):
        X = F.relu(self.f_connected1(X))
        X = F.relu(self.f_connected2(X))
        X = self.out(X)
        return X

# Load the model architecture
model = ANN_Model()

# Load the state dictionary into the model
model.load_state_dict(torch.load('diabetes_state_dict.pt'))
model.eval()

def predict_diabetes(data):
    prediction = model(torch.tensor(data))
    return "Diabetic" if prediction[0] > 0.5 else "Non-diabetic"
    

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = [
            request.form.get('Pregnancies'),
            request.form.get('Glucose'),
            request.form.get('BloodPressure'),
            request.form.get('SkinThickness'),
            request.form.get('Insulin'),
            request.form.get('BMI'),
            request.form.get('DiabetesPedigreeFunction'),
            request.form.get('Age')
        ]
        
        if None in user_input or len(user_input) != 8:
            result = "Please enter valid values for all fields."
        else:
            try:
                user_input = [float(x) for x in user_input]
                result = predict_diabetes(user_input)
            except ValueError:
                result = "Please enter valid numeric values."

        return render_template('index.html', result=result, user_input=user_input)
    
    return render_template('index.html', result=None, user_input=None)

if __name__ == '__main__':
    app.run(debug=True)
