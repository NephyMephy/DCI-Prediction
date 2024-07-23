document.addEventListener('DOMContentLoaded', function() {
    const yearSelect = document.getElementById('year-select');
    const scoresChart = document.getElementById('scoresChart').getContext('2d');
    let chart;

    // Fetch the CSV data
    fetch('/dci_scores.csv')
        .then(response => response.text())
        .then(data => {
            const parsedData = parseCSV(data);
            populateYearSelect(parsedData);
            yearSelect.addEventListener('change', () => updateChart(parsedData));
            updateChart(parsedData); // Initial chart rendering
        })
        .catch(error => console.error('Error fetching CSV data:', error));

    // Parse CSV data
    function parseCSV(data) {
        const rows = data.split('\n').slice(1); // Remove header row
        return rows.map(row => {
            const [year, date, corps, score] = row.split(',');
            return { year, date, corps, score: parseFloat(score) };
        });
    }

    // Populate the year select dropdown

    function populateYearSelect() {
        const yearSelect = document.getElementById('year-select');
        const years = [...new Set(scores.map(score => score.year))]; // Assuming scores is an array of score objects with a 'year' property
    
        years.forEach(year => {
            const option = document.createElement('option');
            option.value = year;
            option.textContent = year;
            yearSelect.appendChild(option);
        });
    }
    
    // Update the chart based on the selected year
    function updateChart(data) {
        const selectedYear = yearSelect.value;
        const filteredData = data.filter(item => item.year === selectedYear);

        const labels = filteredData.map(item => item.date);
        const scores = filteredData.map(item => item.score);

        if (chart) {
            chart.destroy();
        }

        chart = new Chart(scoresChart, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: `Scores for ${selectedYear}`,
                    data: scores,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    fill: false
                }]
            },
            options: {
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day'
                        }
                    },
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
});