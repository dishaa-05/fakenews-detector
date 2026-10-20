# Fake News Detector (ML Mini Project, Phase 3)

Linear SVM + TF-IDF trained on the IFND dataset, deployed with Streamlit.

## Run locally
```
pip install -r requirements.txt
python train_model.py --data IFND.csv     # creates models/linear_svm_pipeline.joblib
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repo (include `models/linear_svm_pipeline.joblib`).
2. Go to share.streamlit.io, choose the repo and set the main file to `app.py`.

## Files
- `app.py` - web application
- `train_model.py` - reproduces the Phase 1/2 pipeline and exports the model
- `models/metrics.json`, `models/eda.json` - results from the Phase 2 notebook, used by the app's charts
- `.streamlit/config.toml` - theme
