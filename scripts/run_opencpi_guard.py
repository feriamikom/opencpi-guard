import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from opencpi_guard.pipeline import run_pipeline

parser = argparse.ArgumentParser(description='Run OpenCPI-Guard on a generic CPI CSV file.')
parser.add_argument('--input', required=True, help='Input CPI panel CSV using common schema')
parser.add_argument('--output', required=True, help='Output directory')
parser.add_argument('--country', default=None, help='Optional country filter')
parser.add_argument('--valid-start', default='2025-06-01')
parser.add_argument('--test-start', default='2025-12-01')
args = parser.parse_args()

metrics = run_pipeline(
    input_csv=args.input,
    output_dir=args.output,
    valid_start=args.valid_start,
    test_start=args.test_start,
    country=args.country,
)
print(metrics.to_string(index=False))
