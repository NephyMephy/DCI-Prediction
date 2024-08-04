from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from datetime import datetime
import main
from main import df, previous_scores

app = Flask(__name__, static_folder='templates', static_url_path='')
CORS(app, origins=["*"])

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        dates_input = request.form.getlist('date[]')
        corps_input = request.form.getlist('corps[]')
        predictions = []
        previous = []
        errors = []
        print(corps_input, dates_input)
        for date_input, corps_name in zip(dates_input, corps_input):
            try:
                date_object = datetime.strptime(date_input, '%Y-%m-%d')
                if corps_name in main.models:
                    predicted_score = main.predict_score(date_object, corps_name, main.models[corps_name])
                    predictions.append((corps_name, date_input, predicted_score))
                    
                else:
                    errors.append(f"Corps name '{corps_name}' not found.")
            except ValueError:
                errors.append("Invalid date format. Please enter a date in the format YYYY-MM-DD.")
            
            # Get previous scores for each date and corps pair
            previous_scores = main.previous_scores(date_input, corps_name)
            previous.append(previous_scores.to_dict(orient='records'))
        
        return jsonify({'predictions': predictions, 'previous': previous, 'errors': errors})
    return render_template('index.html', prediction_made=False)
    
if __name__ == '__main__':
    app.run(debug=True)