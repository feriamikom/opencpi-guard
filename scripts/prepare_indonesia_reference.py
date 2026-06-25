import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from opencpi_guard.adapters.indonesia_bps_final_adapter import load_bps_final_wide

parser = argparse.ArgumentParser(description='Convert prepared BPS wide data into OpenCPI common schema.')
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--include-national', action='store_true')
args = parser.parse_args()

df = load_bps_final_wide(args.input, include_national=args.include_national)
df.to_csv(args.output, index=False)
print(f'Saved {len(df):,} rows to {args.output}')
