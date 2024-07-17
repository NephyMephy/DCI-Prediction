document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    form.addEventListener('submit', function(e) {
        e.preventDefault(); // Prevent the default form submission
        form.classList.add('fade-out'); // Fade out the form

        // Assuming the server expects JSON and returns predictions in JSON format
        const formData = new FormData(form);
        const jsonData = Object.fromEntries(formData.entries());

        fetch('/predict', { // Adjust the URL to where your server expects to receive prediction requests
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(jsonData),
        })
        .then(response => response.json())
        .then(data => {
            displayPredictions(data.predictions); // Assuming the server response includes a predictions field
        })
        .catch(error => console.error('Error:', error));
    });
});

function displayPredictions(predictions) {
    const predictionsContainer = document.getElementById('predictionsContainer');
    predictionsContainer.innerHTML = ''; // Clear existing predictions

    predictions.forEach(prediction => {
        const bubbleElement = document.createElement('div');
        bubbleElement.className = 'predictionBubble';

        const predictionElement = document.createElement('h2');
        predictionElement.className = 'predictionResult';
        predictionElement.textContent = `Predicted Score for ${prediction.corps} on ${prediction.date}: ${prediction.score}`;

        bubbleElement.appendChild(predictionElement);
        predictionsContainer.appendChild(bubbleElement);
    });
}

function removePredictionInput() {
    const predictionInputs = document.getElementById('predictionInputs');
    if (predictionInputs.children.length > 1) { // Ensure at least one prediction set remains
        predictionInputs.removeChild(predictionInputs.lastChild);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const navbarLinks = document.querySelectorAll('.navbar a');
    const currentUrl = window.location.href;

    navbarLinks.forEach(link => {
        if (link.href === currentUrl) {
            link.classList.add('active');
        }
    });
});
