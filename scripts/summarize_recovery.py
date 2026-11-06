import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Write results/recovery/recovery_report.md from the recovery CSVs (works on partial results too).
   python scripts/summarize_recovery.py [recovery_dir]"""
import os
import sys

from src.recovery.report import build_report

d = sys.argv[1] if len(sys.argv) > 1 else "results/recovery"
text = build_report(d)
with open(os.path.join(d, "recovery_report.md"), "w") as f:
    f.write(text)
print(text)
