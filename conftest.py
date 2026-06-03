"""Put the repo root on sys.path so `import versai` works when running pytest."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
