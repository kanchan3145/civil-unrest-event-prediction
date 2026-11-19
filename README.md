# Civil Unrest Event Extraction and Prediction from Social Media Data

An NLP and Machine Learning project for identifying civil-unrest-related information from social-media posts, extracting structured event evidence, analyzing historical trends, and evaluating forecasting models.

## Tech Stack

- Python
- Pandas and NumPy
- Hugging Face Sentence-Transformers (MiniLM)
- Scikit-learn
- PyTorch
- PyArrow and Apache Parquet
- Streamlit
- Machine Learning and Time-Series Forecasting

## Project Features

- Large-scale social-media data preprocessing
- Civil-unrest-related tweet classification
- Structured event evidence extraction
- Temporal trend analysis
- Historical ACLED-based prediction experiments
- Seven-day tweet-volume forecasting using EWMA and moving averages
- Interactive Streamlit dashboard

## Dataset

The project uses historical Twitter data and ACLED event data.

- Raw tweets: 1,084,452
- Processed tweets: 902,401
- Predicted unrest-related tweets: 48,030

Large datasets are excluded from the repository. Dataset access and preparation instructions should be documented separately.

## Machine Learning

The classification pipeline uses pretrained MiniLM sentence embeddings and a trained Logistic Regression classifier.

Historical ACLED-based experiments evaluate event counts, emergence, and escalation.

## Run Locally

Create and activate a Python virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the dashboard:

```powershell
python -m streamlit run src/evaluator_dashboard_full.py
```

**Note:** Some features require locally generated datasets and saved model artifacts that are not included in the GitHub repository.

## Limitations

- Predicted-positive tweets are not verified real-world incidents.
- Event extraction relies partly on rule-based methods.
- Regional geolocation and independent validation require further improvements.
- Historical forecasts should not be interpreted as live unrest predictions.

## Project Status

Research prototype with implemented classification, event extraction, historical forecasting experiments, and interactive visualization.