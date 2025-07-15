from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)  # FIXED

# Load the trained model
model = pickle.load(open('shopper_model.pkl', 'rb'))

# Label encoding maps
month_map = {
    'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
    'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
}
visitor_map = {'Returning_Visitor': 0, 'New_Visitor': 1, 'Other': 2}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        Month = request.form['Month']
        VisitorType = request.form['VisitorType']
        Weekend = 1 if request.form['Weekend'] == 'Yes' else 0

        # Numerical features
        BounceRates = float(request.form['BounceRates'])
        ExitRates = float(request.form['ExitRates'])
        PageValues = float(request.form['PageValues'])
        ProductRelated = int(request.form['ProductRelated'])
        ProductRelated_Duration = float(request.form['ProductRelated_Duration'])
        Administrative = int(request.form['Administrative'])
        Informational = int(request.form['Informational'])

        # Set defaults for missing durations
        Administrative_Duration = 100.0
        Informational_Duration = 50.0

        # Input array
        input_data = pd.DataFrame([[
            Administrative,
            Administrative_Duration,
            Informational,
            Informational_Duration,
            ProductRelated,
            ProductRelated_Duration,
            BounceRates,
            ExitRates,
            PageValues,
            0.0,  # SpecialDay
            month_map.get(Month, 0),
            2,    # OperatingSystems
            2,    # Browser
            1,    # Region
            1,    # TrafficType
            visitor_map.get(VisitorType, 0),
            Weekend
        ]])

        prediction = model.predict(input_data)[0]
        result = "✅ YES, user will make a purchase." if prediction == 1 else "❌ NO, user will not make a purchase."
        return render_template('index.html', prediction_result=result)

    except Exception as e:
        return render_template('index.html', prediction_result=f"Error: {e}")

if __name__ == '__main__':  # FIXED
    app.run(debug=True)
