import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric
import os

DRIFT_SHARE_THRESHOLD = 0.3  # Global variable for drift threshold

def main():
    # Load your datasets
    reference = pd.read_csv("data/raw/movies.csv")
    current = pd.read_csv("data/raw/movies_drift.csv")

    # Create a comprehensive report with several metrics and presets
    report = Report(
        metrics=[
            DataDriftPreset(drift_share=DRIFT_SHARE_THRESHOLD),
            DataQualityPreset(),
            DatasetDriftMetric(drift_share=DRIFT_SHARE_THRESHOLD),
        ]
    )

    report.run(reference_data=reference, current_data=current)

    # Save the report as HTML
    output_path = "reports/drift_report_movies.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # <-- FIX
    report.save_html(output_path)
    print(f"Evidently drift & quality report saved to {output_path}")

if __name__ == "__main__":
    main()