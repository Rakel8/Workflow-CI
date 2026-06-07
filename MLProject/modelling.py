import shutil
import dagshub
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# 1. Inisialisasi DagsHub & MLflow
dagshub.init(repo_owner='Rakel8', repo_name='submission-mlops-dicoding', mlflow=True)
mlflow.set_experiment("Loan_Approval_Tuning_Rakhly")

# 2. Memuat Data Bersih
print("Memuat dataset...")
df = pd.read_csv("loan_data_clean.csv")
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Membagi data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Setup Hyperparameter Tuning (GridSearchCV)
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, None]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')

print("Memulai pelatihan dan Hyperparameter Tuning...")

# --- FIX ERROR DAGSHUB ---
# Menghapus sisa Run ID dari GitHub Actions agar DagsHub membuat Run baru
if "MLFLOW_RUN_ID" in os.environ:
    del os.environ["MLFLOW_RUN_ID"]
# -------------------------

# 4. Memulai MLflow Run (Manual Logging)
with mlflow.start_run():
    # Proses Training
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Prediksi & Kalkulasi Metrik
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # A. Manual Logging (Parameter & Metrik)
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)

    # B. Artefak Tambahan 1: Visualisasi Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # C. Artefak Tambahan 2: Visualisasi Feature Importance
    feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    plt.figure(figsize=(8,6))
    feature_importances.nlargest(10).plot(kind='barh', color='teal')
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    # D. Artefak Tambahan 3: File requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("scikit-learn\npandas\nmatplotlib\nseaborn\nmlflow==2.19.0\ndagshub\n")
    mlflow.log_artifact("requirements.txt")

    # E. Logging Model Utama
    mlflow.sklearn.log_model(best_model, "random_forest_model")

    # Simpan model lokal untuk mempermudah Docker Build di GitHub Actions
    if os.path.exists("local_model"):
        shutil.rmtree("local_model")
    mlflow.sklearn.save_model(best_model, "local_model")
    
    print(f"Selesai! Model terbaik ditemukan dengan akurasi: {acc:.4f}")
    print("Semua metrik dan artefak telah dikirim ke DagsHub.")