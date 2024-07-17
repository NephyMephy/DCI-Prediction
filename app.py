from flask import Flask, request, render_template
from datetime import datetime
import main

app = Flask(__name__, static_folder='templates', static_url_path='')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        dates_input = request.form.getlist('date[]')
        corps_input = request.form.getlist('corps[]')
        predictions = []
        errors = []
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
        return render_template('index.html', predictions=predictions, errors=errors)
    return render_template('index.html')\
    
if __name__ == '__main__':
    app.run(debug=True)