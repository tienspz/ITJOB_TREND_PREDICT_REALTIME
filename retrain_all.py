import os, joblib, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROCESSED_FILE = os.path.join(BASE_DIR, 'data', 'it_jobs_processed.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def execute_retrain():
    import subprocess, sys
    subprocess.run([sys.executable, __file__])

print("[MLOps] FULL RETRAIN ALL 3 MODELS FROM KAGGLE DATA")
print("=" * 55)

df = pd.read_csv(PROCESSED_FILE)
print(f"Loaded {len(df)} rows")

df['domain_seniority'] = df['it_domain'].astype(str) + '_' + df['seniority_level'].astype(str)
df['state_seniority'] = df['state'].astype(str) + '_' + df['seniority_level'].astype(str)

features = ['num_skills', 'skill_diversity', 'skill_programming', 'skill_cloud', 'skill_ai_ml',
            'skill_database', 'skill_devops', 'skill_framework', 'skill_data_engineering',
            'skill_security', 'skill_soft_skills', 'seniority_level', 'job_type', 'state', 'it_domain',
            'domain_seniority', 'state_seniority']

numeric_features = [f for f in features if f.startswith('skill_') or f == 'num_skills']
categorical_features = ['seniority_level', 'job_type', 'state', 'it_domain', 'domain_seniority', 'state_seniority']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# =====================================================================
# 1. SALARY MODEL — try XGBoost first, fallback to tuned RF
# =====================================================================
print("\n[1/3] SALARY MODEL")
salary_df = df.dropna(subset=['salary_annual']).copy()
Q1, Q3 = salary_df['salary_annual'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lo, hi = max(Q1 - 1.5*IQR, 15000), min(Q3 + 1.5*IQR, 500000)
salary_df = salary_df[(salary_df['salary_annual'] >= lo) & (salary_df['salary_annual'] <= hi)]
print(f"  Rows after outlier removal: {len(salary_df)}")

X = salary_df[features]
y = salary_df['salary_annual']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model = None
best_score = -1
best_model_name = ""

try:
    from xgboost import XGBRegressor
    print("  Training XGBoost...")
    pipe_xgb = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1, verbosity=0))
    ])
    pipe_xgb.fit(X_train, y_train)
    r2_xgb = r2_score(y_test, pipe_xgb.predict(X_test))
    mae_xgb = mean_absolute_error(y_test, pipe_xgb.predict(X_test))
    print(f"  XGBoost R²: {r2_xgb:.4f}, MAE: ${mae_xgb:,.0f}")
    best_model = pipe_xgb
    best_score = r2_xgb
    best_model_name = "XGBoost"
except ImportError:
    print("  XGBoost not installed, skipping.")

print("  Training RandomForest with grid search...")
pipe_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_grid = {
    'model__n_estimators': [200, 400],
    'model__max_depth': [15, 25, None],
    'model__min_samples_leaf': [1, 3],
}
gs = GridSearchCV(pipe_rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
r2_rf = r2_score(y_test, gs.predict(X_test))
mae_rf = mean_absolute_error(y_test, gs.predict(X_test))
print(f"  RF tuned R²: {r2_rf:.4f}, MAE: ${mae_rf:,.0f} (params: {gs.best_params_})")

if best_score < r2_rf:
    best_model = gs.best_estimator_
    best_score = r2_rf
    best_model_name = "RF"

print(f"  Best model: {best_model_name} (R²={best_score:.4f})")
joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_salary_model.joblib'))

meta_sal = {
    'feature_names': features,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'mean_salary': float(y.mean()),
    'median_salary': float(y.median()),
    'r2_score': float(best_score),
    'mae': float(mae_xgb if best_model_name == 'XGBoost' else mae_rf),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'model_type': best_model_name,
    'it_domain': sorted(df['it_domain'].dropna().unique().tolist()),
    'seniority_level': sorted(df['seniority_level'].dropna().unique().tolist()),
    'job_type': sorted(df['job_type'].dropna().unique().tolist()),
    'state': sorted(df['state'].dropna().unique().tolist()),
}
joblib.dump(meta_sal, os.path.join(MODELS_DIR, 'salary_model_meta.joblib'))
print("  Saved salary model + meta")

# =====================================================================
# 2. DEMAND MODEL
# =====================================================================
print("\n[2/3] DEMAND MODEL")
demand_df = df.groupby(['it_domain', 'state', 'seniority_level', 'job_type']).size().reset_index(name='posting_count')
demand_df['demand_score'] = np.log1p(demand_df['posting_count'])
max_score = demand_df['demand_score'].max()
demand_df['demand_score'] = (demand_df['demand_score'] / max_score * 100).clip(0, 100)

X_dem = demand_df[['it_domain', 'state', 'seniority_level', 'job_type']]
y_dem = demand_df['demand_score']

preprocessor_dem = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['it_domain', 'state', 'seniority_level', 'job_type'])
])

X_train, X_test, y_train, y_test = train_test_split(X_dem, y_dem, test_size=0.2, random_state=42)

pipe_dem = Pipeline([
    ('preprocessor', preprocessor_dem),
    ('model', RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1))
])
pipe_dem.fit(X_train, y_train)
r2_dem = r2_score(y_test, pipe_dem.predict(X_test))
print(f"  Demand R²: {r2_dem:.4f}")

joblib.dump(pipe_dem, os.path.join(MODELS_DIR, 'demand_model.joblib'))
meta_dem = {
    'model_type': 'RandomForestRegressor (Demand Score 0-100)',
    'r2_score': float(r2_dem),
    'max_posting_count': int(demand_df['posting_count'].max()),
    'it_domain': sorted(df['it_domain'].dropna().unique().tolist()),
    'seniority_level': sorted(df['seniority_level'].dropna().unique().tolist()),
    'job_type': sorted(df['job_type'].dropna().unique().tolist()),
    'state': sorted(df['state'].dropna().unique().tolist()),
}
joblib.dump(meta_dem, os.path.join(MODELS_DIR, 'demand_meta.joblib'))
print("  Saved demand model + meta")

# =====================================================================
# 3. CLUSTER MODEL (KMeans on all features)
# =====================================================================
print("\n[3/3] CLUSTER MODEL")
cluster_df = df.dropna(subset=['salary_annual']).copy()
cluster_df = cluster_df[(cluster_df['salary_annual'] >= lo) & (cluster_df['salary_annual'] <= hi)]
X_cl = cluster_df[features]

pipe_cl = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=5, random_state=42)),
    ('kmeans', KMeans(n_clusters=5, random_state=42, n_init=10))
])
pipe_cl.fit(X_cl)

X_trans = pipe_cl.named_steps['preprocessor'].transform(X_cl)
pca = pipe_cl.named_steps['pca']
X_pca = pca.transform(X_trans)
kmeans = pipe_cl.named_steps['kmeans']
labels = kmeans.labels_

sil = silhouette_score(X_pca, labels)
print(f"  Silhouette score: {sil:.4f}")

cluster_desc = {}
for c in range(5):
    mask = labels == c
    subset = cluster_df[mask]
    top_domain = subset['it_domain'].mode().iloc[0] if len(subset) > 0 else 'N/A'
    top_sen = subset['seniority_level'].mode().iloc[0] if len(subset) > 0 else 'N/A'
    avg_sal = subset['salary_annual'].mean()
    cluster_desc[c] = f"Cluster {c}: {top_domain} / {top_sen} / Avg ${avg_sal:,.0f} ({mask.sum()} jobs)"

joblib.dump(pipe_cl, os.path.join(MODELS_DIR, 'cluster_model.joblib'))
meta_cl = {
    'n_clusters': 5,
    'pca_components': 5,
    'silhouette_score': float(sil),
    'cluster_descriptions': cluster_desc,
    'feature_names': features,
}
joblib.dump(meta_cl, os.path.join(MODELS_DIR, 'cluster_meta.joblib'))
print("  Saved cluster model + meta")

print("\n" + "=" * 55)
print("ALL 3 MODELS RETRAINED SUCCESSFULLY")
print("=" * 55)
