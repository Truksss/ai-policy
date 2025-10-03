from fastapi import APIRouter
from metrics import RAGASMetricsCollector
import json

router = APIRouter()

@router.get("/metrics/summary")
def get_metrics_summary():
    collector = RAGASMetricsCollector()
    return collector.get_metrics_summary()

@router.get("/metrics/raw")
def get_raw_metrics():
    try:
        with open("metrics_log.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "No metrics data found"}

@router.delete("/metrics/clear")
def clear_metrics():
    try:
        with open("metrics_log.json", 'w') as f:
            json.dump({}, f)
        return {"message": "Metrics cleared successfully"}
    except Exception as e:
        return {"error": f"Failed to clear metrics: {str(e)}"}
