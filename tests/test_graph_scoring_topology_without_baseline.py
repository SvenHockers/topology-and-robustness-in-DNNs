import numpy as np


def test_topology_scores_present_when_baseline_disabled_with_dict_config():
    """
    Regression test:
    If `use_baseline_scores` is False we should skip degree/laplacian/etc,
    but we must still compute topology features when `use_topology` is True.

    This specifically covers the case where `graph_params` is passed as a plain dict
    (common in notebooks / serialized configs).
    """
    from src.graph_scoring import compute_graph_scores

    rng = np.random.default_rng(0)
    Z_train = rng.normal(size=(40, 3)).astype(np.float32)
    X_points = rng.normal(size=(5, 3)).astype(np.float32)
    f_train = rng.random(size=(40,)).astype(np.float32)

    graph_params = {
        "use_baseline_scores": False,
        "use_topology": True,
        "space": "input",
        "topo_k": 10,
        "topo_maxdim": 1,
        "topo_min_persistence": 1e-6,
    }

    scores = compute_graph_scores(
        X_points=X_points,
        model=None,  # unused when baseline is off and space is input
        Z_train=Z_train,
        f_train=f_train,
        graph_params=graph_params,
        device="cpu",
    )

    # Baseline keys should be absent
    assert "degree" not in scores
    assert "laplacian" not in scores

    # Topology keys should be present (stable naming from persistence_summary_features)
    assert "topo_h0_count" in scores
    assert "topo_h0_total_persistence" in scores
