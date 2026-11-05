"""
networks.py  (helper module - imported by the other scripts, not run directly)

Network choice for the study:
    source  case14 = PGLib-OPF v23.07  pglib_opf_case14_ieee   (same buses/branches/loads as the dataset,
                                                                 plus PGLib thermal ratings and costs)
    target  case30 = MATPOWER case30 via pandapower.networks.case30()
                     (the exact grid behind the 30-bus dataset; not available in PGLib;
                      keeps its original line ratings)
    extension case57 = PGLib-OPF v23.07 pglib_opf_case57_ieee (larger target grid for the revision;
                     no 2007 time series exists for it)

Also holds the DATASET metadata (MATPOWER nominal Pd per bus and generator buses) used to
turn the dataset's Cons_i columns into per-bus load factors.
"""

import logging
import urllib.request
import warnings
from pathlib import Path

import pandapower.networks as pn
from pandapower.converter.matpower import from_mpc

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)

PROJECT_DIR = Path(__file__).resolve().parents[1]
NETWORK_DIR = PROJECT_DIR / "data" / "networks"
PGLIB_TAG = "v23.07"
PGLIB_URL = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/{tag}/{name}.m"

# Metadata of the grids that generated the hourly dataset (MATPOWER case14 / case30), 1-based buses
DATASET = {
    "case14": dict(
        file="IEEE_14_Bus_Time_Series_Hourly.csv", n_bus=14,
        Pd=[0, 21.7, 94.2, 47.8, 7.6, 11.2, 0, 0, 29.5, 9, 3.5, 6.1, 13.5, 14.9],
        gen_buses=[1, 2, 3, 6, 8],
    ),
    "case30": dict(
        file="IEEE_30_Bus_Time_Series_Hourly.csv", n_bus=30,
        Pd=[0, 21.7, 2.4, 7.6, 0, 0, 22.8, 30, 0, 5.8, 0, 11.2, 0, 6.2, 8.2, 3.5, 9, 3.2, 9.5, 2.2,
            17.5, 0, 3.2, 8.7, 0, 3.5, 0, 0, 2.4, 10.6],
        gen_buses=[1, 2, 13, 22, 23, 27],
    ),
}


def ensure_pglib(name: str) -> Path:
    """Download a PGLib .m file into data\\networks if it is not there yet."""
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    path = NETWORK_DIR / f"{name}.m"
    if not path.exists():
        url = PGLIB_URL.format(tag=PGLIB_TAG, name=name)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def get_net(case: str):
    """Return a fresh pandapower network for 'case14', 'case30' or 'case57'."""
    if case == "case14":
        return from_mpc(str(ensure_pglib("pglib_opf_case14_ieee")))
    if case == "case30":
        return pn.case30()
    if case == "case57":
        return from_mpc(str(ensure_pglib("pglib_opf_case57_ieee")))
    raise ValueError(case)


def load_buses(net):
    """1-based bus number of each row of net.load (same order as net.load)."""
    return (net.load.bus + 1).tolist()
