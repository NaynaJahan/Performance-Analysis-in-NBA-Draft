import pathlib, importlib.util
import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SETS_PATH = ROOT / "src/amla_at1/data/sets.py"
spec = importlib.util.spec_from_file_location("sets_mod", SETS_PATH)
sets_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sets_mod)

def test_pop_target_happy_path():
    Xdf = pd.DataFrame({"employee_id":[1,2,3], "age":[25,33,42], "level":["Jr","Sr","Jr"]})
    y = pd.Series([5,10,20], name="salary")
    df = Xdf.copy(); df["salary"] = y
    X, yy = sets_mod.pop_target(df, "salary")
    pd.testing.assert_frame_equal(X, Xdf)
    pd.testing.assert_series_equal(yy.reset_index(drop=True), y.reset_index(drop=True))

def test_pop_target_missing_col():
    df = pd.DataFrame({"a":[1,2]})
    with pytest.raises(KeyError):
        sets_mod.pop_target(df, "salary")

def test_save_and_load(tmp_path):
    Xtr = np.array([[1,2],[3,4]])
    ytr = np.array([0,1])
    path = tmp_path.as_posix() + "/"
    sets_mod.save_sets(X_train=Xtr, y_train=ytr, path=path)
    X_train, y_train, X_val, y_val, X_test, y_test = sets_mod.load_sets(path)
    assert X_val is None and y_val is None and X_test is None and y_test is None
    assert np.allclose(X_train, Xtr) and np.allclose(y_train, ytr)
