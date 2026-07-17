import os
import subprocess
import sys
import time
from datetime import datetime

import schedule

def run_ml_retrain():
    # Run as a subprocess: importing retrain_all would execute the whole
    # training script at import time (it has no __main__ guard).
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Start ML retrain cycle...")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrain_all.py")
    subprocess.run([sys.executable, script])

def run_realtime_snapshot():
    """Fetch the realtime pipeline once so history_service records a row.

    Builds the project's own longitudinal dataset (data/realtime_history.csv)
    — the Kaggle dataset is a single-week snapshot, so real trends can only
    come from data we accumulate ourselves.
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Realtime snapshot...")
    try:
        from backend.services import trend_service
        report = trend_service.build_realtime_report()
        print(f"  source={report.get('source')}, jobs={report.get('total_jobs')}")
    except Exception as e:
        print(f"  snapshot failed: {e}")

if __name__ == "__main__":
    print("="*60)
    print("MLOPS WORKER STARTED")
    print("="*60)

    # Retrain every Sunday at 2:00 AM
    schedule.every().sunday.at("02:00").do(run_ml_retrain)

    # Accumulate one realtime history snapshot per day (08:00)
    schedule.every().day.at("08:00").do(run_realtime_snapshot)
    run_realtime_snapshot()  # record one immediately on worker start

    print("\nWaiting for schedule. Press Ctrl+C to exit.")
    while True:
        schedule.run_pending()
        time.sleep(60)
