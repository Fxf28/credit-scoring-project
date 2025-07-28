import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

import pandas as pd
import json
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.models import validate_serving_input

# 1. Konfigurasi
experiment_name = "Latihan Credit Scoring"
client = MlflowClient()

# 2. Ambil experiment dan run terakhir
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    raise ValueError(f"Experiment '{experiment_name}' tidak ditemukan.")

experiment_id = experiment.experiment_id
runs = client.search_runs([experiment_id], order_by=["start_time DESC"], max_results=1)
if not runs:
    raise ValueError("Tidak ada run yang ditemukan.")

run_id = runs[0].info.run_id
print("Latest run_id:", run_id)

# 3. URI model
model_uri = f"runs:/{run_id}/model"

# 4. Unduh input example dan validasi
example_json_path = download_artifacts(f"{model_uri}/serving_input_example.json")
with open(example_json_path, "r") as f:
    input_data = json.load(f)

# 5. Validasi input cocok dengan model
validate_serving_input(model_uri, input_data)
print("✅ Serving input berhasil divalidasi terhadap model.")

# 6. Load model dan prediksi
model = mlflow.pyfunc.load_model(model_uri)
df = pd.DataFrame(input_data["dataframe_split"]["data"], columns=input_data["dataframe_split"]["columns"])
pred = model.predict(df)
print("🧠 Prediksi:")
print(pred)

# $env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
# # opsi satu dengan menggunakan run_id
# mlflow models serve -m runs:/<RUN_ID>/<MODEL_PATH> -p <PORT> --env-manager <conda/virtualenv/no-conda> 
# opsi dua dengan menggunakan registry model
# mlflow models serve -m models:/<MODEL_NAME>/<VERSION> -p <PORT> --env-manager <conda/virtualenv/no-conda>
# example
# mlflow models serve -m "models:/credit-scoring/1" --port 5002 --no-conda