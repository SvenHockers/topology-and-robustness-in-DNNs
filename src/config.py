from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import copy
import os

import yaml


@dataclass
class GeneralConfig:
    seed: int = 42
    exp_name: str = "mlp_baseline"
    device: str = "auto"  # "cpu" | "cuda" | "auto"
    output_dir: str = "outputs/robustness/mlp_baseline"
    sample_limit: Optional[int] = None  # limit val samples processed


@dataclass
class DataConfig:
    n_samples_per_shape: int = 200
    n_points: int = 20
    noise: float = 0.1
    val_split: float = 0.2
    batch_size: int = 32


@dataclass
class ModelConfig:
    arch: str = "MLP"  # "MLP" | "CNN"
    train: bool = True
    epochs: int = 20
    lr: float = 1e-3
    checkpoint: Optional[str] = None


@dataclass
class AdversarialProbeConfig:
    enabled: bool = True
    norms: List[str] = field(default_factory=lambda: ["linf", "l2"])
    eps_max: float = 1.0
    steps: int = 40
    tol: float = 1e-3
    outer_bisect: bool = True
    eps_grid: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])


@dataclass
class GeometricRotationConfig:
    axes: List[str] = field(default_factory=lambda: ["z"])  # 'x','y','z'
    deg_range: Tuple[float, float] = (-45.0, 45.0)


@dataclass
class GeometricTranslationConfig:
    axes: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    range: Tuple[float, float] = (-0.2, 0.2)


@dataclass
class GeometricJitterConfig:
    std_range: Tuple[float, float] = (0.0, 0.2)
    clip: float = 0.5


@dataclass
class GeometricDropoutConfig:
    ratio_range: Tuple[float, float] = (0.0, 0.5)


@dataclass
class GeometricProbeConfig:
    enabled: bool = True
    tol: float = 1e-3
    rotation: GeometricRotationConfig = field(default_factory=GeometricRotationConfig)
    translation: GeometricTranslationConfig = field(default_factory=GeometricTranslationConfig)
    jitter: GeometricJitterConfig = field(default_factory=GeometricJitterConfig)
    dropout: GeometricDropoutConfig = field(default_factory=GeometricDropoutConfig)


@dataclass
class InterpolationProbeConfig:
    enabled: bool = True
    pairs_per_class: int = 20
    match: str = "index"  # "index" | "nn"
    steps: int = 50
    cross_class: bool = False


@dataclass
class TopologyProbeConfig:
    enabled: bool = True
    compute_dgm: bool = True
    maxdim: int = 1
    sample_size: int = 200


@dataclass
class LayerwiseTopologyProbeConfig:
    enabled: bool = True
    layers: List[str] = field(default_factory=lambda: ["input", "fc1", "fc2", "fc3", "pooled"])
    distances: List[str] = field(default_factory=lambda: ["wasserstein"])  # "wasserstein", "bottleneck"
    maxdim: int = 1
    sample_size: int = 200
    conditions: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "adv_linf_eps": [0.1, 0.2],
            "adv_l2_eps": [0.1],
            "rotation_deg": [10.0, 20.0],
            "jitter_std": [0.05],
        }
    )


@dataclass
class ReportingConfig:
    save_csv: bool = True
    save_plots: bool = True
    save_artifacts: bool = False
    sample_visualizations_per_class: int = 3


@dataclass
class ProbesConfig:
    adversarial: AdversarialProbeConfig = field(default_factory=AdversarialProbeConfig)
    geometric: GeometricProbeConfig = field(default_factory=GeometricProbeConfig)
    interpolation: InterpolationProbeConfig = field(default_factory=InterpolationProbeConfig)
    topology: TopologyProbeConfig = field(default_factory=TopologyProbeConfig)
    layerwise_topology: LayerwiseTopologyProbeConfig = field(default_factory=LayerwiseTopologyProbeConfig)


@dataclass
class RobustnessConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    probes: ProbesConfig = field(default_factory=ProbesConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    @staticmethod
    def _coerce_types(d: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder for future coercions; currently YAML handles primitives.
        return d

    @classmethod
    def from_yaml(cls, path: str) -> "RobustnessConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        raw = cls._coerce_types(raw)

        # impl helpers because python is dump and doenst handle types like any other proper language :)
        def _ensure_float(d: Dict[str, Any], key: str):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = float(d[key])
                except ValueError:
                    pass

        def _ensure_int(d: Dict[str, Any], key: str):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = int(float(d[key]))
                except ValueError:
                    pass

        def _ensure_bool(d: Dict[str, Any], key: str):
            if key in d and isinstance(d[key], str):
                val = d[key].strip().lower()
                if val in {"true", "1", "yes", "y"}:
                    d[key] = True
                elif val in {"false", "0", "no", "n"}:
                    d[key] = False
        def _ensure_list_float(d: Dict[str, Any], key: str):
            if key in d and isinstance(d[key], list):
                out = []
                for v in d[key]:
                    if isinstance(v, str):
                        try:
                            out.append(float(v))
                        except ValueError:
                            out.append(v)
                    else:
                        out.append(v)
                d[key] = out
        def _ensure_tuple2_float(dct: Dict[str, Any], key: str):
            if key in dct:
                v = dct[key]
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    a, b = v
                    try:
                        a = float(a) if isinstance(a, str) else a
                        b = float(b) if isinstance(b, str) else b
                        dct[key] = (a, b)
                    except ValueError:
                        pass

        def load_section(section_cls, section_key: str, extra: Dict[str, Any] | None = None):
            src = raw.get(section_key, {})
            if extra:
                src = {**src, **extra}
            return section_cls(**src)

        if "general" in raw and isinstance(raw["general"], dict):
            general = load_section(GeneralConfig, "general")
        else:
            top_keys = {k: raw[k] for k in ["seed", "exp_name", "device", "output_dir", "sample_limit"] if k in raw}
            general = GeneralConfig(**top_keys)
        # Data section with safe coercions
        data_src = dict(raw.get("data", {}))
        _ensure_int(data_src, "n_samples_per_shape")
        _ensure_int(data_src, "n_points")
        _ensure_float(data_src, "noise")
        _ensure_float(data_src, "val_split")
        _ensure_int(data_src, "batch_size")
        data = DataConfig(**data_src)

        # Model section with safe coercions
        model_src = dict(raw.get("model", {}))
        _ensure_bool(model_src, "train")
        _ensure_int(model_src, "epochs")
        _ensure_float(model_src, "lr")
        model = ModelConfig(**model_src)
        # Probes (with type coercion)
        probes_raw = raw.get("probes", {})
        # Adversarial
        adv_src = dict(probes_raw.get("adversarial", {}))
        _ensure_float(adv_src, "eps_max")
        _ensure_int(adv_src, "steps")
        _ensure_float(adv_src, "tol")
        _ensure_list_float(adv_src, "eps_grid")
        _ensure_bool(adv_src, "enabled")
        _ensure_bool(adv_src, "outer_bisect")
        adversarial = AdversarialProbeConfig(**adv_src)

        # Geometric
        geo_src = dict(probes_raw.get("geometric", {}))
        _ensure_bool(geo_src, "enabled")
        _ensure_float(geo_src, "tol")
        geometric = GeometricProbeConfig(
            **{k: v for k, v in geo_src.items() if k not in {"rotation", "translation", "jitter", "dropout"}}
        )
        # nested geometry configs
        if "rotation" in geo_src:
            rot_src = dict(geo_src["rotation"])
            _ensure_tuple2_float(rot_src, "deg_range")
            geometric.rotation = GeometricRotationConfig(**rot_src)
        if "translation" in geo_src:
            tr_src = dict(geo_src["translation"])
            _ensure_tuple2_float(tr_src, "range")
            geometric.translation = GeometricTranslationConfig(**tr_src)
        if "jitter" in geo_src:
            jit_src = dict(geo_src["jitter"])
            _ensure_tuple2_float(jit_src, "std_range")
            _ensure_float(jit_src, "clip")
            geometric.jitter = GeometricJitterConfig(**jit_src)
        if "dropout" in geo_src:
            dr_src = dict(geo_src["dropout"])
            _ensure_tuple2_float(dr_src, "ratio_range")
            geometric.dropout = GeometricDropoutConfig(**dr_src)

        # Interpolation
        inter_src = dict(probes_raw.get("interpolation", {}))
        _ensure_bool(inter_src, "enabled")
        _ensure_int(inter_src, "pairs_per_class")
        _ensure_int(inter_src, "steps")
        _ensure_bool(inter_src, "cross_class")
        interpolation = InterpolationProbeConfig(**inter_src)

        # Topology
        topo_src = dict(probes_raw.get("topology", {}))
        _ensure_bool(topo_src, "enabled")
        _ensure_bool(topo_src, "compute_dgm")
        _ensure_int(topo_src, "maxdim")
        _ensure_int(topo_src, "sample_size")
        topology_cfg = TopologyProbeConfig(**topo_src)

        # Layerwise topology
        ltopo_src = dict(probes_raw.get("layerwise_topology", {}))
        _ensure_bool(ltopo_src, "enabled")
        _ensure_int(ltopo_src, "maxdim")
        _ensure_int(ltopo_src, "sample_size")
        # conditions: dict of lists -> coerce to floats
        if "conditions" in ltopo_src and isinstance(ltopo_src["conditions"], dict):
            cond = {}
            for k, v in ltopo_src["conditions"].items():
                if isinstance(v, list):
                    new_list = []
                    for item in v:
                        try:
                            new_list.append(float(item))
                        except Exception:
                            new_list.append(item)
                    cond[k] = new_list
                else:
                    cond[k] = v
            ltopo_src["conditions"] = cond
        layerwise_topology = LayerwiseTopologyProbeConfig(**ltopo_src)

        probes = ProbesConfig(
            adversarial=adversarial,
            geometric=geometric,
            interpolation=interpolation,
            topology=topology_cfg,
            layerwise_topology=layerwise_topology,
        )

        reporting = load_section(ReportingConfig, "reporting")

        cfg = cls(general=general, data=data, model=model, probes=probes, reporting=reporting)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        # General
        assert self.general.device in {"auto", "cpu", "cuda"}
        assert self.data.val_split > 0 and self.data.val_split < 1
        assert self.data.n_points > 0 and self.data.n_samples_per_shape > 0
        assert self.model.arch in {"MLP", "CNN"}
        # Probes validation
        for norm in self.probes.adversarial.norms:
            assert norm in {"linf", "l2"}
        for ax in self.probes.geometric.rotation.axes:
            assert ax in {"x", "y", "z"}
        for ax in self.probes.geometric.translation.axes:
            assert ax in {"x", "y", "z"}
        for dist in self.probes.layerwise_topology.distances:
            assert dist in {"wasserstein", "bottleneck"}

    def resolved_output_dir(self) -> str:
        return os.path.join(self.general.output_dir)


