import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import pandas as pd
from opencpi_guard.schema import coerce_schema, validate_schema

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--strict', action='store_true')
args = parser.parse_args()

df = pd.read_csv(args.input)
df = coerce_schema(df)
report = validate_schema(df, strict=args.strict)
print(json.dumps(report, indent=2, default=str))
if not report['is_valid']:
    raise SystemExit(1)
