import numpy as np

def extract_basic_features(cycle):
    """
    cycle: dict with keys 'voltage_V', 'current_A', 'charge_capacity', 'discharge_capacity'
    Returns: feature dict
    """
    V, I = np.array(cycle["voltage_V"]), np.array(cycle["current_A"])
    cap_chg, cap_dis = cycle["charge_capacity"], cycle["discharge_capacity"]

    features = {
        "charge_capacity": cap_chg,
        "discharge_capacity": cap_dis,
        "coulombic_efficiency": cap_dis / cap_chg if cap_chg > 0 else 0.0,
        "mean_voltage": np.mean(V),
        "var_voltage": np.var(V),
    }
    return features

