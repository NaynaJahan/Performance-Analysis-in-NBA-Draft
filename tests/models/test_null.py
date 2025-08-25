import pathlib, importlib.util
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = ROOT / "src/amla_at1/models/null.py"
spec = importlib.util.spec_from_file_location("null_mod", PATH)
null_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(null_mod)

def test_null_model():
    X = pd.DataFrame({"x":[1,2,3,4]})
    y = pd.Series([0,1,1,0])
    m = null_mod.NullModel().fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (4,2)
    assert np.allclose(proba[:,1], y.mean())
