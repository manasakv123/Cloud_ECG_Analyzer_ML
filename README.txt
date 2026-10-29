CLOUD ECG ANALYZER WITH RANDOM FOREST CLASSIFIER

1. Replace your old app.py with this app.py, or extract this ZIP to a new folder.
2. In the activated virtual environment, run:
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
3. Upload the original labelled ECG CSV (not the 140-row sample conversion).
4. Open the "Heartbeat classifier" tab and click "Train and evaluate Random Forest".

Assumption: numeric feature columns come first and the final column contains the class label.
Check the dataset documentation before interpreting results. This is educational, not diagnostic.
