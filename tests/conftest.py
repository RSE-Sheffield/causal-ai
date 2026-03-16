"""Shared fixtures for causal-ai tests."""

import json

import pytest


# ---------------------------------------------------------------------------
# Sample CTF artifacts matching the real schema
# ---------------------------------------------------------------------------

SAMPLE_DAG_DOT = """\
digraph dag {
    rankdir=LR;
    fp_precision -> gpu_memory_peak_mb;
    batch_size -> training_time_seconds;
    seed;
}
"""

SAMPLE_TESTS = {
    "tests": [
        {
            "name": "seed _||_ gpu_memory_peak_mb",
            "estimator": "LinearRegressionEstimator",
            "estimate_type": "coefficient",
            "effect": "direct",
            "treatment_variable": "seed",
            "formula": "gpu_memory_peak_mb ~ seed",
            "alpha": 0.05,
            "skip": False,
            "expected_effect": {"gpu_memory_peak_mb": "NoEffect"},
        },
        {
            "name": "fp_precision --> gpu_memory_peak_mb",
            "estimator": "LinearRegressionEstimator",
            "estimate_type": "coefficient",
            "effect": "direct",
            "treatment_variable": "fp_precision",
            "formula": "gpu_memory_peak_mb ~ fp_precision",
            "alpha": 0.05,
            "skip": False,
            "expected_effect": {"gpu_memory_peak_mb": "SomeEffect"},
        },
    ]
}

SAMPLE_RESULTS = [
    {
        "name": "seed _||_ gpu_memory_peak_mb",
        "estimate_type": "coefficient",
        "effect": "direct",
        "treatment_variable": "seed",
        "expected_effect": {"gpu_memory_peak_mb": "NoEffect"},
        "formula": "gpu_memory_peak_mb ~ seed",
        "alpha": 0.05,
        "skip": False,
        "passed": True,
        "result": {
            "treatment": "seed",
            "outcome": "gpu_memory_peak_mb",
            "adjustment_set": [],
            "effect_measure": "coefficient",
            "effect_estimate": {"seed": -4.05e-05},
            "ci_low": {"seed": -0.0157},
            "ci_high": {"seed": 0.0156},
        },
    },
    {
        "name": "fp_precision --> gpu_memory_peak_mb",
        "estimate_type": "coefficient",
        "effect": "direct",
        "treatment_variable": "fp_precision",
        "expected_effect": {"gpu_memory_peak_mb": "SomeEffect"},
        "formula": "gpu_memory_peak_mb ~ fp_precision",
        "alpha": 0.05,
        "skip": False,
        "passed": True,
        "result": {
            "treatment": "fp_precision",
            "outcome": "gpu_memory_peak_mb",
            "adjustment_set": [],
            "effect_measure": "coefficient",
            "effect_estimate": {"fp_precision[T.fp32-true]": 93.32},
            "ci_low": {"fp_precision[T.fp32-true]": 59.24},
            "ci_high": {"fp_precision[T.fp32-true]": 127.40},
        },
    },
    {
        "name": "batch_size --> training_time_seconds",
        "estimate_type": "coefficient",
        "effect": "direct",
        "treatment_variable": "batch_size",
        "expected_effect": {"training_time_seconds": "SomeEffect"},
        "formula": "training_time_seconds ~ batch_size",
        "alpha": 0.05,
        "skip": False,
        "passed": False,
        "result": {
            "treatment": "batch_size",
            "outcome": "training_time_seconds",
            "adjustment_set": [],
            "effect_measure": "coefficient",
            "effect_estimate": {"batch_size": -0.239},
            "ci_low": {"batch_size": -0.258},
            "ci_high": {"batch_size": -0.220},
        },
    },
]

SAMPLE_RUNTIME_CSV = """\
learning_rate,batch_size,optimiser_type,adaptation_method,seed,fp_precision,training_time_seconds,gpu_memory_peak_mb,test_target_acc,test_source_acc
0.0003,64,AdamW,DAN,1,tf32,255.99,131.90,0.957,0.994
0.0003,64,AdamW,DAN,7,tf32,250.98,131.90,0.951,0.990
0.001,128,SGD,DANN,42,fp16-mixed,200.12,38.57,0.920,0.985
0.001,256,SGD,CDAN,123,fp32-true,180.45,225.10,0.935,0.992
"""


@pytest.fixture
def cluster_dir(tmp_path):
    """Create a temporary cluster directory with all CTF artifacts."""
    d = tmp_path / "stanage"
    d.mkdir()

    (d / "dag.dot").write_text(SAMPLE_DAG_DOT)
    (d / "causal_tests_stanage.json").write_text(json.dumps(SAMPLE_TESTS))
    (d / "causal_test_results_stanage.json").write_text(json.dumps(SAMPLE_RESULTS))
    (d / "runtime_data_stanage.csv").write_text(SAMPLE_RUNTIME_CSV)

    return d


@pytest.fixture
def multi_cluster_dir(tmp_path):
    """Create a parent directory with two cluster subdirectories."""
    for name in ("stanage", "bede"):
        d = tmp_path / name
        d.mkdir()
        (d / "dag.dot").write_text(SAMPLE_DAG_DOT)
        (d / f"causal_tests_{name}.json").write_text(json.dumps(SAMPLE_TESTS))
        (d / f"runtime_data_{name}.csv").write_text(SAMPLE_RUNTIME_CSV)

    # Stanage: all pass except one
    (tmp_path / "stanage" / "causal_test_results_stanage.json").write_text(
        json.dumps(SAMPLE_RESULTS)
    )

    # Bede: flip the third test to passed
    bede_results = json.loads(json.dumps(SAMPLE_RESULTS))
    bede_results[2]["passed"] = True
    (tmp_path / "bede" / "causal_test_results_bede.json").write_text(
        json.dumps(bede_results)
    )

    return tmp_path
