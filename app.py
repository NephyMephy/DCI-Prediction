from flask import Flask, request, render_template
from datetime import datetime
import main 

app = Flask(__name__, static_folder='templates', static_url_path='')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        date_input = request.form['date']
        corps_input = request.form['corps']
        try:
            date_object = datetime.strptime(date_input, '%Y-%m-%d')
            if corps_input in main.models:
                predicted_score = main.predict_score(date_object, corps_input, main.models[corps_input])
                return render_template('index.html', prediction=predicted_score, corps=corps_input, date=date_input)
            else:
                return render_template('index.html', error=f"Corps name '{corps_input}' not found.")
        except ValueError:
            return render_template('index.html', error="Invalid date format. Please enter a date in the format YYYY-MM-DD.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)