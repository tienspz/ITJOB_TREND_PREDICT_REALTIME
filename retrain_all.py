import os, joblib, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score
from sklearn.inspection import permutation_importance

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
            'skill_security', 'skill_soft_skills', 'years_experience',
            'seniority_level', 'job_type', 'state', 'it_domain',
            'domain_seniority', 'state_seniority']

numeric_features = [f for f in features if f.startswith('skill_') or f in ('num_skills', 'years_experience')]
categorical_features = ['seniority_level', 'job_type', 'state', 'it_domain', 'domain_seniority', 'state_seniority']

# years_experience is only extractable from ~45% of postings -> median-impute
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), numeric_features),
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

# 80/20 split. The 20% test set is ISOLATED: it is never used for training,
# tuning, or model selection — only for ONE final evaluation of the winner.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
CV_FOLDS = 3

# Candidate models are compared by cross-validation ON THE TRAIN SET ONLY.
candidates = {}  # name -> {'cv_mean', 'cv_std', 'model'}

try:
    from xgboost import XGBRegressor
    print(f"  Training XGBoost ({CV_FOLDS}-fold CV on train set)...")
    pipe_xgb = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1, verbosity=0))
    ])
    cv_xgb = cross_val_score(pipe_xgb, X_train, y_train, cv=CV_FOLDS, scoring='r2', n_jobs=-1)
    pipe_xgb.fit(X_train, y_train)
    candidates['XGBoost'] = {'cv_mean': cv_xgb.mean(), 'cv_std': cv_xgb.std(), 'model': pipe_xgb}
    print(f"  XGBoost CV R²: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}")
except ImportError:
    print("  XGBoost not installed, skipping.")

print(f"  Training RandomForest (GridSearchCV, {CV_FOLDS}-fold on train set)...")
pipe_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_grid = {
    'model__n_estimators': [200, 400],
    'model__max_depth': [15, 25, None],
    'model__min_samples_leaf': [1, 3],
}
gs = GridSearchCV(pipe_rf, param_grid, cv=CV_FOLDS, scoring='r2', n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
candidates['RF'] = {
    'cv_mean': gs.best_score_,
    'cv_std': float(gs.cv_results_['std_test_score'][gs.best_index_]),
    'model': gs.best_estimator_,
}
print(f"  RF tuned CV R²: {gs.best_score_:.4f} (params: {gs.best_params_})")

# Select the winner by train-set CV score (test set untouched so far)
best_model_name = max(candidates, key=lambda k: candidates[k]['cv_mean'])
best = candidates[best_model_name]
best_model = best['model']
print(f"  Best by CV: {best_model_name} (CV R²={best['cv_mean']:.4f} ± {best['cv_std']:.4f})")

# FINAL evaluation — the one and only use of the isolated 20% test set
y_pred = best_model.predict(X_test)
r2_test = r2_score(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)
print(f"  FINAL (isolated test 20%): R²={r2_test:.4f}, MAE=${mae_test:,.0f}")

# Permutation importance on the test set (explains the final model without
# touching training decisions — computed after model selection is locked)
print("  Computing permutation importance (test set)...")
perm = permutation_importance(best_model, X_test, y_test, n_repeats=5,
                              random_state=42, scoring='r2', n_jobs=-1)
importance = sorted(
    ({'feature': f, 'importance': round(float(m), 5)}
     for f, m in zip(features, perm.importances_mean)),
    key=lambda r: r['importance'], reverse=True)
for row in importance[:5]:
    print(f"    {row['feature']:<22} {row['importance']:.4f}")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    top = importance[:12][::-1]
    plt.figure(figsize=(9, 6))
    plt.barh([r['feature'] for r in top], [r['importance'] for r in top],
             color='#00d2ff', alpha=0.7)
    plt.xlabel('Permutation importance (drop in R², test set)')
    plt.title('Salary Model — Feature Importance')
    plt.tight_layout()
    fig_dir = os.path.join(BASE_DIR, 'reports', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved feature_importance.png")
except Exception as e:
    print(f"  (figure skipped: {e})")

joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_salary_model.joblib'), compress=3)

meta_sal = {
    'feature_names': features,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'mean_salary': float(y.mean()),
    'median_salary': float(y.median()),
    'r2_score': float(r2_test),
    'mae': float(mae_test),
    'cv_r2_mean': float(best['cv_mean']),
    'cv_r2_std': float(best['cv_std']),
    'cv_folds': CV_FOLDS,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'model_type': best_model_name,
    'feature_importance': importance[:12],
    'salary_coverage': {
        'rows_total': int(len(df)),
        'rows_with_salary': int(df['salary_annual'].notna().sum()),
        'rows_range_based': int(df['salary_min'].notna().sum()) if 'salary_min' in df.columns else 0,
        'rows_with_yoe': int(df['years_experience'].notna().sum()) if 'years_experience' in df.columns else 0,
    },
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

# Same protocol: 80/20 split, CV on train, one final test evaluation
X_train, X_test, y_train, y_test = train_test_split(X_dem, y_dem, test_size=0.2, random_state=42)

pipe_dem = Pipeline([
    ('preprocessor', preprocessor_dem),
    ('model', RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1))
])
cv_dem = cross_val_score(pipe_dem, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
print(f"  Demand CV R² (5-fold, train): {cv_dem.mean():.4f} ± {cv_dem.std():.4f}")
pipe_dem.fit(X_train, y_train)
r2_dem = r2_score(y_test, pipe_dem.predict(X_test))
print(f"  Demand FINAL (isolated test 20%): R²={r2_dem:.4f}")

joblib.dump(pipe_dem, os.path.join(MODELS_DIR, 'demand_model.joblib'), compress=3)
meta_dem = {
    'model_type': 'RandomForestRegressor (Demand Score 0-100)',
    'r2_score': float(r2_dem),
    'cv_r2_mean': float(cv_dem.mean()),
    'cv_r2_std': float(cv_dem.std()),
    'cv_folds': 5,
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

joblib.dump(pipe_cl, os.path.join(MODELS_DIR, 'cluster_model.joblib'), compress=3)
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
