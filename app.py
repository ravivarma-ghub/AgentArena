import os
import sys
from pathlib import Path

# Forward to ui/dashboard.py
dashboard_path = Path(__file__).resolve().parent / "ui" / "dashboard.py"
with open(dashboard_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(code, globals())
