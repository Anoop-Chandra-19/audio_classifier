# Audio Classifier

## Project Overview
The Audio Classifier project is a machine learning pipeline designed to classify audio tracks into various genres. It leverages the Free Music Archive (FMA) dataset and processes audio files into log-mel spectrograms for training and evaluation. The project is still under development, with additional scripts and features planned for future implementation.

## Features
- **Audio Preprocessing**: Converts raw audio files into log-mel spectrograms using `librosa` and saves them as PyTorch tensors.
- **Custom Dataset**: A PyTorch `Dataset` class for loading spectrograms and their corresponding labels.
- **Metadata Handling**: Reads and processes metadata from the FMA dataset.

## Installation

### Prerequisites
- Python 3.13 or higher
- Linux operating system
- [uv](https://github.com/astral-sh/uv) package manager (for dependency management)

### Dependencies
Install the required dependencies using the following command:
```bash
uv pip install -r pyproject.toml
```
Or, to add a new dependency:
```bash
uv pip install <package-name>
```

All dependencies are managed via `pyproject.toml` and `uv.lock`. You do not need a `requirements.txt` file.

## Usage

### Preprocessing Audio Files
To preprocess audio files and generate log-mel spectrograms, run the `preprocess.py` script:
```bash
uv run python -m scripts.preprocess
```
This script processes audio files in the `data/raw/fma_small` directory and saves the spectrograms in the `data/processed` directory.

### Loading the Dataset
The `AudioDataset` class in `src/data_loader.py` can be used to load the processed spectrograms and their corresponding labels:
```python
from src.data_loader import AudioDataset

dataset = AudioDataset(
    processed_data_dir='data/processed',
    metadata_file='data/fma_metadata/tracks.csv'
)
print(f"Number of samples: {len(dataset)}")
spec, label = dataset[0]
print(f"Sample spec shape: {spec.shape}, Sample label: {label}")
```

## Project Structure
```
audio_classifier
├── main.py
├── pyproject.toml
├── README.md
├── uv.lock
├── data/
│   ├── fma_metadata.zip
│   ├── fma_metadata/
│   │   ├── tracks.csv
│   │   └── ...
│   ├── processed/
│   │   ├── 000002.pt
│   │   └── ...
│   └── raw/
├── scripts/
│   └── preprocess.py
└── src/
    ├── data_loader.py
    ├── utils.py
    └── ...
```

## Future Work
- Implement additional scripts for model training and evaluation.
- Add support for more audio formats and datasets.
- Develop a FastAPI-based web interface for real-time audio classification.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The Free Music Archive (FMA) dataset for providing the audio tracks and metadata.
- The developers of `librosa`, `PyTorch`, and other open-source libraries used in this project.