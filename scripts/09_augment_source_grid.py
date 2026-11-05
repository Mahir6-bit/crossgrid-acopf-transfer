"""
09_augment_source_grid.py
Step 9: data augmentation of the source grid (IEEE 14-bus): pools L, LP and LPT.
The code is in augment_case14.py (a shared module, because 12_make_n1_sets.py imports its functions);
this file runs it with the same command-line options, e.g.
    python scripts\\09_augment_source_grid.py
"""

import runpy
import sys
from pathlib import Path

sys.argv[0] = str(Path(__file__).with_name("augment_case14.py"))
runpy.run_path(sys.argv[0], run_name="__main__")
