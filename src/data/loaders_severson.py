import pandas as pd
import os

def load_severson(path: str, rated_capacity=1.1):
    """
    Load Severson 2019 dataset (LFP 18650 cells).
    Args:
        path: path to the CSV or JSON files
        rated_capacity: nominal cell capacity (Ah)
    Returns:
        DataFrame with one row per cycle
    """
    # Example: using BEEP preprocessed data (structured JSON)
    all_cycles = []
    for fname in os.listdir(path):
        if fname.endswith(".json"):
            df = pd.read_json(os.path.join(path, fname))
            for cycle in df["cycles"]:
                cap = cycle["discharge_capacity"]
                soh = cap / rated_capacity * 100
                all_cycles.append({
                    "cell_id": fname.split(".")[0],
                    "cycle_index": cycle["cycle_index"],
                    "soh": soh,
                    "charge_capacity": cycle["charge_capacity"],
                    "discharge_capacity": cap,
                    "time_s": cycle["time"],
                    "voltage_V": cycle["voltage"],
                    "current_A": cycle["current"],
                })
    return pd.DataFrame(all_cycles)

