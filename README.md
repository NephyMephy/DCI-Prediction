# DCI-Prediction
A machine learning model to predict future Drum Corps International (DCI) scores based on historical data.

## Features

- Predicts DCI scores for a specified corps on a given date
- Uses historical data including season, date, corps name, and previous scores
- Implements data preprocessing, feature engineering, and machine learning techniques
- PLANNED: Clean Easy UI for a simple interface to interact with the prediction algorithm.
    - Current: Easy to Use CLI-based system. 
- Open Class Corps Historical Data from 2014 and onwards can be used due to scoring regulations changing
    - Additionally, Open Class Corps data of DCI World Class Competitions has been omitted if the score difference between Open Class Finals and World Class Prelims exceeds 3.

## Corps Status
 - Blue Devils A: 2023-current
 - Blue Devils B: 2022-current

Planned:
Mandarins
Carolina Crown
Bluecoats
Boston Crusaders
Phantom Regiment

## Usage
1. Clone the Repo
2. Install all needed Requirements
3. Run program.py in your desired IDE
4. When the Graph pops up, close it and return to terminal
5. Follow instruction in the Terminal


## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn (AKA sklearn)
- matplotlib
- [Additional libraries depending on the chosen model, e.g., statsmodels for ARIMA, prophet, or tensorflow for LSTM]


Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Drum Corps International for inspiring this project
- Boredom
- Github Copilot for fixing my stupid code
