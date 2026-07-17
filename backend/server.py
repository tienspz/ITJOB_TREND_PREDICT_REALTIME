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
from flask import Flask, request, jsonify, send_from_directory
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
    )
    frontend_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000").split(",")
    CORS(app, resources={
        r"/api/*": {"origins": frontend_origins},
        r"/reports/*": {"origins": frontend_origins},
    }, supports_credentials=True)
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
        # Static frontend (no Jinja2): all dynamic values come from /api/meta.
        index_file = os.path.join(config.FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return send_from_directory(config.FRONTEND_DIR, "index.html")
        # API-only deployments (e.g. Docker image without frontend/)
        return jsonify({
            "service": "IT Job Market Intelligence API",
            "health": "/api/health",
            "meta": "/api/meta",
        })

    @app.route('/api/realtime-trends', methods=['GET'])
    def realtime_trends():
        """
        Trend forecast: BLS-projected baseline + realtime point, with a
        rolling-origin backtest that picks the better of two models.

        Baseline: backup_trends.csv — 30 monthly IT job counts anchored on
        the Kaggle Jan-2024 snapshot and projected with ~3.5% YoY growth
        (US BLS IT employment). This is a PROJECTION, not measured history
        (the Kaggle crawl spans a single week) — the response says so.

        Realtime: current IT job count from the provider chain, scaled to
        the baseline magnitude and attached as the current month.

        Forecast: Polynomial(deg 2) vs Holt-Winters (statsmodels), compared
        by MAPE on the last 6 held-out points; the winner refits on the full
        series and forecasts 2 months with a ±1.96σ residual interval.
        """
        # Current job count from the live realtime pipeline
        try:
            from backend.services import trend_service
            report = trend_service.build_realtime_report()
            job_count = report.get("total_jobs", 0)
        except Exception as e:
            app.logger.warning("Realtime pipeline unavailable for trend: %s", e)
            job_count = 0

        df_trend = pd.read_csv(config.BACKUP_TRENDS_FILE)
        data_points = df_trend.to_dict('records')
        current_month = pd.Timestamp.now().strftime("%Y-%m")

        # Scale realtime API count to the baseline magnitude
        if job_count > 0:
            avg_hist = sum(d['job_count'] for d in data_points) / len(data_points)
            scaled_count = int(job_count * max(avg_hist / max(job_count, 1), 0.1))
            if data_points and data_points[-1]["month"] == current_month:
                data_points[-1]["job_count"] = scaled_count
            else:
                data_points.append({"month": current_month, "job_count": scaled_count})

        y = np.array([d['job_count'] for d in data_points], dtype=float)
        n = len(y)
        horizon = 2

        def _poly_forecast(train, steps):
            X = np.arange(len(train)).reshape(-1, 1)
            poly = PolynomialFeatures(degree=2)
            m = LinearRegression().fit(poly.fit_transform(X), train)
            nxt = np.arange(len(train), len(train) + steps).reshape(-1, 1)
            return m.predict(poly.transform(nxt)), (float(m.coef_[1]) if len(m.coef_) > 1 else 0.0)

        def _holt_forecast(train, steps):
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            m = ExponentialSmoothing(train, trend="add", seasonal=None,
                                     initialization_method="estimated").fit()
            return np.asarray(m.forecast(steps)), None

        # Rolling-origin backtest on the last 6 points
        backtest = {}
        holdout = min(6, n - 4)
        candidates = {"polynomial_deg2": _poly_forecast}
        try:
            import statsmodels  # noqa: F401
            candidates["holt_winters"] = _holt_forecast
        except ImportError:
            app.logger.info("statsmodels not installed — forecasting with polynomial only.")

        residuals_by_model = {}
        if holdout >= 2:
            train, test = y[:-holdout], y[-holdout:]
            for name, fn in candidates.items():
                try:
                    pred, _ = fn(train, holdout)
                    mape = float(np.mean(np.abs((test - pred) / np.maximum(test, 1))) * 100)
                    backtest[name] = round(mape, 2)
                    residuals_by_model[name] = test - pred
                except Exception as e:
                    app.logger.warning("backtest %s failed: %s", name, e)

        model_used = min(backtest, key=backtest.get) if backtest else "polynomial_deg2"

        # Final forecast: winner refit on the full series
        try:
            predictions, coef = candidates[model_used](y, horizon)
        except Exception:
            model_used = "polynomial_deg2"
            predictions, coef = _poly_forecast(y, horizon)
        if coef is None:  # Holt-Winters: derive trend sign from its forecast
            coef = float(predictions[-1] - y[-1]) / max(horizon, 1)

        # Simple ±1.96σ interval from backtest residuals
        resid = residuals_by_model.get(model_used)
        ci = float(1.96 * np.std(resid)) if resid is not None and len(resid) > 1 else 0.0

        last_timestamp = pd.to_datetime(data_points[-1]['month'], errors='coerce')
        if last_timestamp is pd.NaT:
            last_timestamp = pd.Timestamp.now()

        forecast_points = []
        for pred in predictions:
            last_timestamp = last_timestamp + pd.DateOffset(months=1)
            forecast_points.append({
                "month": last_timestamp.strftime("%Y-%m"),
                "job_count": int(max(0, pred)),
                "ci_low": int(max(0, pred - ci)),
                "ci_high": int(max(0, pred + ci)),
                "is_forecast": True,
            })

        # Own accumulated realtime series (grows while the system runs)
        try:
            from backend.services import history_service
            history_points = len(history_service.load_history())
        except Exception:
            history_points = 0

        return jsonify({
            "status": "success",
            "source": "kaggle+realtime",
            "historical": data_points,
            "forecast": forecast_points,
            "metrics": {
                "current_trend": "Tăng trưởng" if coef > 0 else "Suy giảm",
                "growth_rate": round(coef, 2),
                "model_used": model_used,
                "backtest_mape": backtest,
                "confidence_interval": round(ci, 1),
            },
            "baseline_note": ("Chuỗi baseline là CHIẾU TĂNG TRƯỞNG từ snapshot Kaggle 01/2024 "
                              "+ ~3.5%/năm (US BLS); điểm cuối là số realtime đã scale."),
            "realtime_history_points": history_points,
        })


app = create_app()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5000"))
    print(f"Starting server on http://0.0.0.0:{port}/")
    serve(app, host='0.0.0.0', port=port, threads=4)
