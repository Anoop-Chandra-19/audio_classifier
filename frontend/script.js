const form = document.getElementById('uploadForm');
const fileIN = document.getElementById('fileInput');
const statusel = document.getElementById('status');
const results = document.getElementById('results');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    results.innerHTML = '';
    statusel.textContent = 'Uploading & Classifying...';

    const file = fileIN.files[0];
    if (!file) {
        statusel.textContent = 'Please select an audio file.';
        return;
    }

    const data = new FormData();
    data.append('file', file);

    try {
        const res = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: data,
        });
        
        if (!res.ok) {
            throw new Error(`Server error: ${res.status}`);
        }

        const json = await res.json();
        statusel.textContent = '';

        json.predictions.array.forEach(({genre, confidence}) => {
            const div = document.createElement('div');
            div.className = 'prediction';
            div.innerHTML = `
                <span>${genre}</span>
                <span>${(confidence * 100).toFixed(1)}%</span>`
        });
    } catch (err){
        statusel.textContent = 'Error: ' + err.message;
        console.error(err);
    }
});