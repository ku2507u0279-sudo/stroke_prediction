# Stroke Risk Prediction — Precision-Tuned Version

A simple Streamlit demonstration with five inputs: age, hypertension, heart disease, average glucose level, and BMI.

## Files
- `app.py` — Streamlit UI
- `stroke_model.joblib` — compact trained model
- `threshold.json` — decision threshold selected using validation data
- `train_stroke_model.ipynb` — Google Colab notebook to retrain
- `requirements.txt` — dependencies
- `training_metrics.json` — validation-selected threshold and held-out test metrics

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Upload `app.py`, `stroke_model.joblib`, `threshold.json`, and `requirements.txt` to GitHub. Deploy `app.py` through Streamlit Community Cloud.

## Retrain in Colab
Open `train_stroke_model.ipynb`, run all cells, upload the original CSV when prompted, then download `stroke_model.joblib` and `threshold.json`.

## Threshold tuning
The notebook makes train/validation/test splits. It chooses a decision threshold on validation data to maximize precision while retaining at least 50% recall where possible. Final metrics are reported on the held-out test set. This may increase precision but typically lowers recall; there is no guaranteed precision target.

## Medical disclaimer
This is an educational prototype, not a clinically validated screening or diagnostic system. Model scores are not calibrated clinical probabilities. The dataset's label prevalence and quality affect results; never use these predictions for real medical decisions.
