"""
Flask Backend Application
=========================

Entry point. Creates the Flask app, loads ML models once at startup into the
shared model registry, registers the blueprint routers, and serves the
frontend dashboard.

Original responsibilities (preserved):
  * /api/health, /api/meta, /api/predict_salary, /api/predict_demand,
    /api/cluster, /api/upload_cv  ->  moved into routers/
  * /api/realtime-trends  ->  kept here; the frontend dashboard "Trend"
    button consumes its exact {status, source, historical, forecast, metrics}
    contract. Now uses Kaggle-derived monthly counts (30 months) + realtime
    API current count as the Linear Regression training set.

New (lecturer-mandated realtime pipeline):
  * /api/realtime/*, /api/trending, /api/realtime-jobs  ->  routers/realtime.py
"""

import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from waitress import serve

# Ensure project root is on sys.path when running directly as a script
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend import config
from backend import model_registry
from backend.routers import core as core_router
from backend.routers import predict as predict_router
from backend.routers import realtime as realtime_router
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
except ImportError:  # pragma: no cover
    LinearRegression = None
    PolynomialFeatures = None

try:
    from pytrends.request import TrendReq
    HAS_GOOGLE_TRENDS = True
except ImportError:
    TrendReq = None
    HAS_GOOGLE_TRENDS = False


def create_app():
    app = Flask(
        __name__,
        static_folder=os.path.join(config.FRONTEND_DIR, "static"),
        template_folder=os.path.join(config.FRONTEND_DIR, "templates"),
    )
    frontend_origins = os.environ.get("FRONTEND_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000").split(",")
    CORS(app, resources={r"/api/*": {"origins": frontend_origins}}, supports_credentials=True)
    app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_BYTES

    # Load ML models once into the shared registry.
    model_registry.load_all(logger=app.logger)

    app.register_blueprint(core_router.bp)
    app.register_blueprint(predict_router.bp)
    app.register_blueprint(realtime_router.bp)

    _register_legacy_routes(app)
    return app


def _register_legacy_routes(app):
    """Routes whose HTML/behaviour is tightly coupled to server startup."""

    @app.route('/')
    def index():
        salary_meta = model_registry.get_salary_meta()
        demand_meta = model_registry.get_demand_meta()
        cluster_meta = model_registry.get_cluster_meta()
        # Compute live stat overrides
        data_file = config.PROCESSED_DATA_FILE
        ds_size = 0
        top_domain = "—"
        top_skill = "—"
        if os.path.exists(data_file):
            import pandas as pd
            try:
                df_v = pd.read_csv(data_file, usecols=["it_domain", "skill_programming", "skill_cloud", "skill_ai_ml", "skill_database"])
                ds_size = len(df_v)
                top_domain = df_v["it_domain"].value_counts().index[0]
                skill_cols = ["skill_programming", "skill_cloud", "skill_ai_ml", "skill_database"]
                top_skill = df_v[skill_cols].sum().idxmax().replace("skill_", "").title()
            except Exception:
                pass
        return render_template("index.html", meta=salary_meta, demand_meta=demand_meta, cluster_meta=cluster_meta, ds_size=ds_size, top_domain=top_domain, top_skill=top_skill)

    @app.route('/api/realtime-trends', methods=['GET'])
    def realtime_trends():
        """
        Trend forecast using Kaggle-derived historical data + realtime API.
        
        Historical: Monthly IT job counts from Kaggle LinkedIn data (Jan 2024
        as baseline) projected with ~3.5% YoY growth (US BLS IT employment).
        Stored in backup_trends.csv (30 months).
        
        Realtime: Current IT job count from the live provider chain
        (freehire.dev -> RemoteOK -> cache -> snapshot).
        
        Forecast: Polynomial regression (degree 2) over the combined series.
        """
        data_points = []
        source_label = "kaggle+realtime"
        backup_file = config.BACKUP_TRENDS_FILE

        # Get current job count from the live realtime pipeline
        try:
            from backend.services import trend_service
            report = trend_service.build_realtime_report()
            job_count = report.get("total_jobs", 0)
            source_label = "kaggle+realtime"
        except Exception as e:
            app.logger.warning("Realtime pipeline unavailable for trend: %s", e)
            job_count = 0

        df_trend = pd.read_csv(backup_file)
        data_points = df_trend.to_dict('records')
        current_month = pd.Timestamp.now().strftime("%Y-%m")

        # Scale realtime API count to match Kaggle magnitude for meaningful chart
        if job_count > 0:
            avg_hist = sum(d['job_count'] for d in data_points) / len(data_points)
            scale_factor = max(avg_hist / max(job_count, 1), 0.1)
            scaled_count = int(job_count * scale_factor)
            if data_points and data_points[-1]["month"] == current_month:
                data_points[-1]["job_count"] = scaled_count
            else:
                data_points.append({"month": current_month, "job_count": scaled_count})

        # Polynomial regression (degree 2) for curved trend forecast
        X = np.array(range(len(data_points))).reshape(-1, 1)
        y = np.array([d['job_count'] for d in data_points])

        if LinearRegression is not None and PolynomialFeatures is not None and len(data_points) >= 3:
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            next_indices = np.array([len(data_points), len(data_points) + 1]).reshape(-1, 1)
            predictions = model.predict(poly.transform(next_indices))
            coef = float(model.coef_[1]) if len(model.coef_) > 1 else 0.0
        elif LinearRegression is not None and len(data_points) >= 2:
            model = LinearRegression()
            model.fit(X, y)
            next_indices = np.array([len(data_points), len(data_points) + 1]).reshape(-1, 1)
            predictions = model.predict(next_indices)
            coef = float(model.coef_[0])
        else:
            predictions = [y[-1], y[-1]]
            coef = 0.0

        last_point_label = data_points[-1]['month']
        last_timestamp = pd.to_datetime(last_point_label, errors='coerce')
        if last_timestamp is pd.NaT:
            last_timestamp = pd.Timestamp.now()

        forecast_points = []
        for pred in predictions:
            if source_label == 'google_trends':
                last_timestamp = last_timestamp + pd.Timedelta(days=1)
                forecast_label = last_timestamp.strftime("%Y-%m-%d")
            else:
                last_timestamp = last_timestamp + pd.DateOffset(months=1)
                forecast_label = last_timestamp.strftime("%Y-%m")
            forecast_points.append({
                "month": forecast_label,
                "job_count": int(max(0, pred)),
                "is_forecast": True,
            })

        return jsonify({
            "status": "success",
            "source": source_label,
            "historical": data_points,
            "forecast": forecast_points,
            "metrics": {
                "current_trend": "Tăng trưởng" if coef > 0 else "Suy giảm",
                "growth_rate": round(coef, 2),
            },
        })


app = create_app()


if __name__ == '__main__':
    print("Starting server on http://0.0.0.0:5000/")
    serve(app, host='0.0.0.0', port=5000, threads=4)
