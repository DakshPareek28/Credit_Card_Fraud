import joblib
import pandas as pd
import os
from django.shortcuts import render
from django.conf import settings

# Load model once when server starts
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fraud_detection_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'amount_scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature order must match training - V1 to V28 + Amount_scaled
FEATURE_COLUMNS = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled']

def home(request):
    result = None
    
    if request.method == 'POST':
        try:
            # Get V1-V28 from form
            v_values = {f'V{i}': float(request.POST.get(f'v{i}', 0)) for i in range(1, 29)}
            amount = float(request.POST.get('amount', 0))
            
            # Scale amount
            amount_scaled = scaler.transform([[amount]])[0][0]
            
            # Build input dataframe in correct column order
            input_data = {**v_values, 'Amount_scaled': amount_scaled}
            input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
            
            prob = model.predict_proba(input_df)[0][1]
            flag = "FRAUD" if prob > 0.5 else "LEGIT"
            
            result = {
                'probability': round(prob * 100, 2),
                'flag': flag
            }
        except Exception as e:
            result = {'error': str(e)}
    
    return render(request, 'detector/home.html', {'result': result})