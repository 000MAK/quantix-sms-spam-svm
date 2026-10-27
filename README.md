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

## Results

Local run with the defaults (seed 42), held-out test set of 1,115 messages: accuracy **0.989**, spam precision **0.979**, spam recall **0.940**, spam F1 **0.959**, ROC-AUC **0.992**. Spam is the minority class, so read recall and F1 rather than accuracy.

## Repository contents

| File | Purpose |
|---|---|
| `.gitignore` | Keeps data, weights and outputs out of Git |
| `LICENSE` | MIT licence for the code |
| `main.py` | Training and evaluation script |
| `requirements.txt` | Python dependencies |

## License

Code is released under the MIT License (see `LICENSE`). Datasets keep their own licences and are downloaded at run time, not redistributed here.

## Related

Part of a set of compact, reproducible ML/DL baselines by [000MAK](https://github.com/000MAK). Flagship research project: [QMASE martial-arts analytics engine](https://github.com/000MAK/quantix-martial-arts-engine).
