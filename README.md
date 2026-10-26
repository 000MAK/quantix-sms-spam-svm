# Quantix SMS Spam SVM

End-to-end train, validation, and test example using a TF-IDF + linear SVM pipeline.

## Dataset and attribution

The script downloads the UCI SMS Spam Collection to `data/raw/` on first run (about 467 KB). It contains 5,574 labeled messages. The dataset is licensed under CC BY 4.0; attribution: Almeida, T. & Hidalgo, J. (2011), [SMS Spam Collection, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84).

The data and fitted artifacts are excluded from Git.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

The deterministic split is stratified: 64% train, 16% validation, and 20% held-out test. Metrics and a serialized model are written to `outputs/`.
