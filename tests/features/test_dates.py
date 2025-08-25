import pathlib, importlib.util
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = ROOT / "src/amla_at1/features/dates.py"
spec = importlib.util.spec_from_file_location("dates_mod", PATH)
dates_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dates_mod)

def test_parse_height():
    assert dates_mod.parse_height_to_inches("6-10") == 82.0
    assert dates_mod.parse_height_to_inches("6'10\"") == 82.0
    assert dates_mod.parse_height_to_inches("6") == 72.0
    assert dates_mod.parse_height_to_inches("6-Jun") == 82.0  # heuristic

def test_add_domain_features():
    df = pd.DataFrame({
        "yr":["Fr","Jr"], "type":["R","P"], "FTM":[5,0],"FTA":[10,0],
        "twoPM":[4,3],"twoPA":[8,6], "TPM":[2,0],"TPA":[4,0],
        "pts":[10,0], "mp":[20,1], "ht":["6-10","6-5"]
    })
    out = dates_mod.add_domain_features(df)
    for c in ["yr_ordinal","is_regular","is_post","ft_ratio","twoP_ratio","tp_ratio","pts_per_min","height_in"]:
        assert c in out.columns
