# Iris Classifier API

A FastAPI service that serves a scikit-learn model trained on the classic Iris dataset. Deployed on: https://iris-prediction-mor5.onrender.com/ 

## What the model predicts

Given four flower measurements in centimetres (sepal length, sepal width, petal length, petal width), the model predicts the **iris species**. It returns one of three classes:

| class_index | species |
|---|---|
| 0 | setosa |
| 1 | versicolor |
| 2 | virginica |

The response also includes the predicted probability for each species.

The model is a scikit-learn `Pipeline` (StandardScaler + classifier). It is saved together with its metadata in `artifacts/iris_model.pkl`. It is trained on only 150 samples, so it will return a confident answer even for measurements that are unlike any real iris. Treat it as a demo, not a botanical tool.

## Project structure

```
iris-api/
├── iris_ml_pipeline.ipynb   # EDA, training, exports the .pkl and CSVs
├── main.py                  # FastAPI app
├── static/
│   └── index.html           # browser UI served at /
├── requirements.txt
├── README.md
└── artifacts/
    ├── iris_model.pkl       # created by the notebook
    ├── iris_raw.csv
    └── iris_processed.csv
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Browser UI for checking health and making predictions |
| GET | `/health` | Service status and whether the model loaded |
| POST | `/predict` | Predicts the species from the four measurements |

### Example request body for `/predict`

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Example response

```json
{
  "species": "setosa",
  "class_index": 0,
  "probabilities": {
    "setosa": 0.9794,
    "versicolor": 0.0119,
    "virginica": 0.0087
  }
}
```

All four fields are required and must be greater than 0 and less than 20. Invalid input returns `422`. If the model file failed to load, `/predict` returns `503`, and `/health` shows `"model_loaded": false` with the error.

## Run locally

### 1. Create and activate a virtual environment

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install jupyter matplotlib seaborn   # only needed to run the notebook
```

### 3. Generate the model (skip if `artifacts/iris_model.pkl` already exists)

```bash
jupyter nbconvert --to notebook --execute iris_ml_pipeline.ipynb --inplace
```

You can also open the notebook in Jupyter and choose Run All. It writes the `.pkl` and CSVs into `artifacts/`.

> The `scikit-learn` version in `requirements.txt` must match the version used to train the `.pkl`. If you retrain under a different version, update the pin.

### 4. Start the API

```bash
uvicorn main:app --reload
```

### 5. Test it

Open the UI at http://127.0.0.1:8000/ or the interactive docs at http://127.0.0.1:8000/docs, or use curl:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

On Windows PowerShell, use `curl.exe` instead of `curl`. You may need to escape the JSON quotes, or use the `/docs` page instead.

### 6. Deactivate when done

```bash
deactivate
```
