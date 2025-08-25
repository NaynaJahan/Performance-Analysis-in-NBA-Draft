import pathlib, importlib.util
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = ROOT / "src/amla_at1/models/performance.py"
spec = importlib.util.spec_from_file_location("perf_mod", PATH)
perf_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf_mod)

def test_metrics_from_proba():
    y = np.array([0,1,1,0,1])
    p = np.array([0.1,0.8,0.7,0.2,0.9])
    m = perf_mod.metrics_from_proba(p, y)
    assert "auroc" in m and "brier" in m
    assert 0.0 <= m["auroc"] <= 1.0

def test_weighted_blend():
    a = np.array([0.2, 0.8])
    b = np.array([0.4, 0.6])
    out = perf_mod.weighted_blend([a,b], [1.0, 1.0])
    assert np.allclose(out, np.array([0.3, 0.7]))
