import schedule
import time
from datetime import datetime
from retrain_all import execute_retrain

def run_ml_retrain():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Start ML retrain cycle...")
    execute_retrain()

if __name__ == "__main__":
    print("="*60)
    print("MLOPS WORKER STARTED")
    print("="*60)

    # Retrain every Sunday at 2:00 AM
    schedule.every().sunday.at("02:00").do(run_ml_retrain)

    print("\nWaiting for schedule. Press Ctrl+C to exit.")
    while True:
        schedule.run_pending()
        time.sleep(60)
