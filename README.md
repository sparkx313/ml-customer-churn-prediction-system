# Customer Churn Prediction

A local, file-driven training pipeline and FastAPI inference service. The trainer learns from a labeled CSV, evaluates a stratified holdout split, saves a scikit-learn pipeline, and serves predictions from that artifact. It does not include fabricated data or fixed probabilities.

## Install and train

Python 3.10 or newer is recommended.

~~~sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python server.py train --data customers.csv --target Churn --model churn_model.joblib
~~~

The target must have exactly two classes. Positive labels such as Yes, True, 1, or Churn are inferred when present; otherwise the lexically last label is used. Set --positive-label to make the choice explicit. IDs and other leakage-prone columns can be excluded with --drop-columns CustomerID,AccountNumber.

Training uses a stratified 80/20 holdout, median/mode imputation, one-hot encoding, standard scaling for numeric fields, and class-weighted logistic regression. The printed metrics include ROC-AUC, precision, recall, F1, accuracy, and a confusion matrix. Review the threshold and metrics for the intended business use; a test split is not a guarantee of future performance.

## Run the API

~~~sh
python server.py serve --model churn_model.joblib --host 127.0.0.1 --port 8000
~~~

The API exposes GET /health and POST /predict. Example request body:

~~~json
{
  "features": {
    "tenure": 36,
    "monthly_charges": 65.5,
    "contract": "Month-to-month"
  }
}
~~~

Feature names must match the training CSV exactly. The model artifact is loaded from a trusted local file; do not load joblib files from untrusted sources. The API returns a churn probability, the configured threshold, and the predicted target label.

## Docker

~~~sh
docker build -t customer-churn .
docker run --rm -v "$PWD:/data" customer-churn train --data /data/customers.csv --target Churn --model /data/churn_model.joblib
docker run --rm -p 8000:8000 -v "$PWD:/data" customer-churn serve --host 0.0.0.0 --model /data/churn_model.joblib
~~~

Customer data and model decisions can affect people. Check data authorization, feature leakage, subgroup performance, and appropriate human review before using predictions operationally.