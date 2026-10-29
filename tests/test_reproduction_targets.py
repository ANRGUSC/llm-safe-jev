from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jev_repro.protocol import JEV_MODEL, MODEL_ID
from jev_repro.reproduction_targets import QWEN_REVISION, JEV_MODEL_ID
from jev_repro.run_experiment import run
import inspect


def test_historical_protocol_not_overwritten():
    assert JEV_MODEL == 'jev-latest'
    assert MODEL_ID == 'Qwen/Qwen2.5-1.5B'


def test_forward_pins_are_distinct_and_exposed():
    assert QWEN_REVISION == '8faed761d45a263340a0528343f099c05c9a4323'
    assert JEV_MODEL_ID == 'jev-1.14.0'
    sig = inspect.signature(run)
    assert sig.parameters['model_revision'].default == QWEN_REVISION
    assert sig.parameters['jev_model'].default == JEV_MODEL_ID


def test_bsd_license_and_human_audit_documented():
    root=Path(__file__).resolve().parents[1]
    assert (root/'LICENSE').read_text().startswith('BSD 3-Clause License')
    text=(root/'docs/AUDIT_PROVENANCE.md').read_text()
    assert 'personally inspecting each of the 13' in text
    assert 'not a second independent blinded annotation study' in text
