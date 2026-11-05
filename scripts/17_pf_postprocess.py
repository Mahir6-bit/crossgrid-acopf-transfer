"""
17_pf_postprocess.py
Step 17 (E6): AC power-flow correction of the predictions, feasibility check and DC-OPF baseline.
The code is in pf_postprocess.py (a shared module, because 18_pf_breakdown_sensitivity.py imports its functions);
this file runs it with the same command-line options, e.g.
    python scripts\\17_pf_postprocess.py
"""

import runpy
import sys
from pathlib import Path

sys.argv[0] = str(Path(__file__).with_name("pf_postprocess.py"))
runpy.run_path(sys.argv[0], run_name="__main__")
