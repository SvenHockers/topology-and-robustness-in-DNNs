from __future__ import annotations

import os
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import RobustnessConfig
from ..data import GiottoPointCloudDataset, make_point_clouds
from ..models import SimplePointMLP, SimplePointCNN
from ..training import train_one_epoch, evaluate
from .probes import (
    estimate_min_eps,
    robust_accuracy_curve,
    find_geometric_flip_threshold,
    search_interpolation_boundary,
    compare_layer_topology,
    pgd_attack,
)
from .report import (
    SampleRecord,
    LayerwiseRecord,
    DiagramDistanceRecord,
    RobustnessSummary,
    write_csv,
    save_curve_png,
    save_hist_png,
    save_layer_distance_bar,
)
from ..topology import compute_layer_topology, extract_persistence_stats


class RobustnessPipeline:
    def __init__(self, config: RobustnessConfig, output_dir: str):
        self.cfg = config
        self.out_dir = output_dir
        self.device = torch.device("cuda" if (self.cfg.general.device == "cuda" or (self.cfg.general.device == "auto" and torch.cuda.is_available())) else "cpu")
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._model: nn.Module | None = None

        # seeds
        torch.manual_seed(self.cfg.general.seed)
        np.random.seed(self.cfg.general.seed)

    def prepare_data(self):
        d = self.cfg.data
        pcs, labels = make_point_clouds(d.n_samples_per_shape, d.n_points, d.noise)
        # keep raw for optional sample visualizations
        self._raw_point_clouds = pcs
        self._raw_labels = labels
        n_total = len(labels)
        n_train = int((1.0 - d.val_split) * n_total)
        idx = np.random.permutation(n_total)
        train_idx, val_idx = idx[:n_train], idx[n_train:]
        train_ds = GiottoPointCloudDataset(pcs[train_idx], labels[train_idx])
        val_ds = GiottoPointCloudDataset(pcs[val_idx], labels[val_idx])
        self._train_loader = DataLoader(train_ds, batch_size=d.batch_size, shuffle=True)
        self._val_loader = DataLoader(val_ds, batch_size=d.batch_size, shuffle=False)
        # optional: visualize sample diagrams per class
        if self.cfg.reporting.save_plots and self.cfg.reporting.sample_visualizations_per_class:
            try:
                from ..visualization import visualize_sample_diagrams
                import shutil as _shutil
                save_path = os.path.join(self.out_dir, "persistence_diagrams_by_class.png")
                visualize_sample_diagrams(
                    self._raw_point_clouds,
                    self._raw_labels,
                    n_samples_per_class=int(self.cfg.reporting.sample_visualizations_per_class),
                    maxdim=int(self.cfg.probes.topology.maxdim) if hasattr(self.cfg.probes, "topology") else 2,
                    save_path=save_path,
                    show=False,
                    seed=int(self.cfg.general.seed),
                )
                src_path = save_path
                if os.path.exists(src_path): # not sure what we shoyuld do here...
                    pass
            except Exception as _e:
                pass

    def prepare_model(self):
        m = self.cfg.model
        if m.arch == "MLP":
            model = SimplePointMLP(num_classes=3)
        else:
            model = SimplePointCNN(num_classes=3)
        self._model = model.to(self.device)

        if m.checkpoint and os.path.exists(m.checkpoint):
            state = torch.load(m.checkpoint, map_location=self.device)
            self._model.load_state_dict(state)
            self._model.eval()
            return

        if m.train:
            optimizer = torch.optim.Adam(self._model.parameters(), lr=m.lr)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(1, m.epochs + 1):
                train_one_epoch(self._model, self._train_loader, optimizer, criterion, self.device)
            # quick val to warm-up
            evaluate(self._model, self._val_loader, criterion, self.device)
            # save checkpoint
            torch.save(self._model.state_dict(), os.path.join(self.out_dir, "model.pth"))
        else:
            self._model.eval()

    def run_probes(self):
        assert self._model is not None and self._val_loader is not None
        model = self._model
        cfg = self.cfg
        device = self.device

        sample_records: List[SampleRecord] = []
        layerwise_records: List[LayerwiseRecord] = []
        diagdist_records: List[DiagramDistanceRecord] = []

        # robust accuracy curves
        ra_curves: Dict[str, List[float]] = {}
        if cfg.probes.adversarial.enabled and cfg.probes.adversarial.eps_grid:
            for norm in cfg.probes.adversarial.norms:
                ra = robust_accuracy_curve(model, self._val_loader, norm, cfg.probes.adversarial.eps_grid, cfg.probes.adversarial.steps)
                ra_curves[norm] = ra

        # per-sample probes
        model.eval()
        sample_id = 0
        for batch_idx, (x, y) in enumerate(self._val_loader):
            x, y = x.to(device), y.to(device)
            for i in range(x.size(0)):
                if cfg.general.sample_limit is not None and sample_id >= cfg.general.sample_limit:
                    break
                xi = x[i].detach().cpu()
                yi = y[i].detach().cpu()
                # compute and cache clean layerwise stats once
                clean_layer_stats: Dict[str, Dict[str, float]] = {}
                with torch.no_grad():
                    _ = model(x[i : i + 1], save_layers=True)
                    layers = cfg.probes.layerwise_topology.layers if cfg.probes.layerwise_topology.enabled else list(model.layer_outputs.keys())
                    for layer in layers:
                        act = model.layer_outputs[layer]
                        dgm = None
                        try:
                            dgm = compute_layer_topology(
                                act, sample_size=cfg.probes.layerwise_topology.sample_size if cfg.probes.layerwise_topology.enabled else 200,
                                maxdim=cfg.probes.layerwise_topology.maxdim if cfg.probes.layerwise_topology.enabled else 1
                            )
                        except Exception:
                            dgm = None
                        if dgm is not None:
                            stats = extract_persistence_stats(dgm)
                            clean_layer_stats[layer] = {f"clean_{k}": float(v) for k, v in stats.items()}
                with torch.no_grad():
                    logits = model(x[i : i + 1], save_layers=False)
                    pred = int(logits.argmax(1).item())
                    # margin = logit_true - best_other
                    logits_np = logits.squeeze(0).detach().cpu().numpy()
                    true_logit = logits_np[int(yi.item())]
                    best_other = np.max(np.delete(logits_np, int(yi.item())))
                    margin = float(true_logit - best_other)

                rec = SampleRecord(
                    sample_id=sample_id,
                    true_label=int(yi.item()),
                    clean_pred=pred,
                    clean_margin=margin,
                )

                # adversarial min eps (linf, l2)
                if cfg.probes.adversarial.enabled:
                    for norm in cfg.probes.adversarial.norms:
                        eps_star, x_adv = estimate_min_eps(
                            model,
                            xi,
                            yi,
                            norm=norm,
                            eps_max=cfg.probes.adversarial.eps_max,
                            steps=cfg.probes.adversarial.steps,
                            tol=cfg.probes.adversarial.tol,
                        )
                        if norm == "linf":
                            rec.eps_star_linf = eps_star
                            if x_adv is not None:
                                with torch.no_grad():
                                    pred_adv = int(model(x_adv.to(device), save_layers=False).argmax(1).item())
                                rec.adv_pred_linf = pred_adv
                        else:
                            rec.eps_star_l2 = eps_star
                            if x_adv is not None:
                                with torch.no_grad():
                                    pred_adv = int(model(x_adv.to(device), save_layers=False).argmax(1).item())
                                rec.adv_pred_l2 = pred_adv

                        # layerwise topology at configured eps (optional, with clean cache)
                        if cfg.probes.layerwise_topology.enabled:
                            layers = cfg.probes.layerwise_topology.layers
                            for eps_target in cfg.probes.layerwise_topology.conditions.get(f"adv_{norm}_eps", []):
                                x_adv_t = pgd_attack(
                                    model, xi.unsqueeze(0).to(device), yi.to(device),
                                    norm=norm, eps=eps_target, steps=cfg.probes.adversarial.steps, step_frac=1.0, random_start=True
                                ).detach().cpu()
                                with torch.no_grad():
                                    _ = model(x_adv_t.to(device), save_layers=True)
                                    for layer in layers:
                                        act_alt = model.layer_outputs[layer]
                                        dgm_alt = compute_layer_topology(
                                            act_alt,
                                            sample_size=cfg.probes.layerwise_topology.sample_size,
                                            maxdim=cfg.probes.layerwise_topology.maxdim,
                                            normalize=cfg.probes.topology.normalize,
                                            pca_dim=cfg.probes.topology.pca_dim,
                                            bootstrap_repeats=cfg.probes.topology.bootstrap_repeats,
                                        )
                                        if dgm_alt is None:
                                            continue
                                        # write clean stats from cache
                                        for H in [0, 1]:
                                            stats_clean = clean_layer_stats.get(layer, {})
                                            layerwise_records.append(
                                                LayerwiseRecord(
                                                    sample_id=sample_id,
                                                    condition=f"adv_{norm}_eps={eps_target}",
                                                    layer=layer,
                                                    betti=f"H{H}",
                                                    count=float(stats_clean.get(f"clean_H{H}_count", 0.0)),
                                                    mean_persistence=float(stats_clean.get(f"clean_H{H}_mean_persistence", 0.0)),
                                                    max_persistence=float(stats_clean.get(f"clean_H{H}_max_persistence", 0.0)),
                                                    total_persistence=float(stats_clean.get(f"clean_H{H}_total_persistence", 0.0)),
                                                )
                                            )
                                        # distances vs clean
                                        # reuse diagram distance via compare helper for consistency
                                        topo = compare_layer_topology(
                                            model,
                                            xi.unsqueeze(0).to(device),
                                            x_adv_t.to(device),
                                            device,
                                            layers=[layer],
                                            maxdim=cfg.probes.layerwise_topology.maxdim,
                                            sample_size=cfg.probes.layerwise_topology.sample_size,
                                            distances=cfg.probes.layerwise_topology.distances,
                                            normalize=cfg.probes.topology.normalize,
                                            pca_dim=cfg.probes.topology.pca_dim,
                                            bootstrap_repeats=cfg.probes.topology.bootstrap_repeats,
                                        )
                                        metrics = topo.get(layer, {})
                                        for dist_name in cfg.probes.layerwise_topology.distances:
                                            for H in [0, 1]:
                                                key = f"{dist_name}_H{H}"
                                                if key in metrics:
                                                    diagdist_records.append(
                                                        DiagramDistanceRecord(
                                                            sample_id=sample_id,
                                                            condition=f"adv_{norm}_eps={eps_target}",
                                                            layer=layer,
                                                            metric=dist_name,
                                                            H=H,
                                                            distance=float(metrics[key]),
                                                        )
                                                    )
                            # noise floor per layer (clean vs clean subsamples)
                            if cfg.probes.layerwise_topology.enabled:
                                layers = cfg.probes.layerwise_topology.layers
                                with torch.no_grad():
                                    _ = model(x[i : i + 1], save_layers=True)
                                    for layer in layers:
                                        act = model.layer_outputs[layer]
                                        dgm_a = compute_layer_topology(
                                            act,
                                            sample_size=cfg.probes.layerwise_topology.sample_size,
                                            maxdim=cfg.probes.layerwise_topology.maxdim,
                                            normalize=cfg.probes.topology.normalize,
                                            pca_dim=cfg.probes.topology.pca_dim,
                                            bootstrap_repeats=1,
                                        )
                                        dgm_b = compute_layer_topology(
                                            act,
                                            sample_size=cfg.probes.layerwise_topology.sample_size,
                                            maxdim=cfg.probes.layerwise_topology.maxdim,
                                            normalize=cfg.probes.topology.normalize,
                                            pca_dim=cfg.probes.topology.pca_dim,
                                            bootstrap_repeats=1,
                                        )
                                        if dgm_a is None or dgm_b is None:
                                            continue
                                        # compute wasserstein H0/H1 as noise floor
                                        from persim import wasserstein
                                        for H in [0, 1]:
                                            A = dgm_a[H] if len(dgm_a) > H else None
                                            B = dgm_b[H] if len(dgm_b) > H else None
                                            if A is None or B is None:
                                                continue
                                            A = A[np.isfinite(A[:, 1])]
                                            B = B[np.isfinite(B[:, 1])]
                                            if len(A) == 0 and len(B) == 0:
                                                dist_val = 0.0
                                            elif len(A) == 0 or len(B) == 0:
                                                continue
                                            else:
                                                dist_val = float(wasserstein(A, B, matching=False))
                                            diagdist_records.append(
                                                DiagramDistanceRecord(
                                                    sample_id=sample_id,
                                                    condition="noise_floor",
                                                    layer=layer,
                                                    metric="wasserstein",
                                                    H=H,
                                                    distance=dist_val,
                                                )
                                            )

                # geometric thresholds (per axis where applicable)
                if cfg.probes.geometric.enabled:
                    # rotation
                    for ax in cfg.probes.geometric.rotation.axes:
                        theta, _ = find_geometric_flip_threshold(
                            model,
                            xi,
                            yi,
                            family="rotation",
                            axis=ax,
                            param_range=cfg.probes.geometric.rotation.deg_range,
                            tol=cfg.probes.geometric.tol,
                        )
                        if ax == "z":
                            rec.rot_deg_star = theta if theta is not None else rec.rot_deg_star
                    # translation
                    for ax in cfg.probes.geometric.translation.axes:
                        t_star, _ = find_geometric_flip_threshold(
                            model,
                            xi,
                            yi,
                            family="translation",
                            axis=ax,
                            param_range=cfg.probes.geometric.translation.range,
                            tol=cfg.probes.geometric.tol,
                        )
                        if ax == "x":
                            rec.trans_x_star = t_star
                        elif ax == "y":
                            rec.trans_y_star = t_star
                        elif ax == "z":
                            rec.trans_z_star = t_star
                    # jitter
                    j_star, _ = find_geometric_flip_threshold(
                        model,
                        xi,
                        yi,
                        family="jitter",
                        axis=None,
                        param_range=cfg.probes.geometric.jitter.std_range,
                        tol=cfg.probes.geometric.tol,
                    )
                    rec.jitter_std_star = j_star
                    # dropout
                    d_star, _ = find_geometric_flip_threshold(
                        model,
                        xi,
                        yi,
                        family="dropout",
                        axis=None,
                        param_range=cfg.probes.geometric.dropout.ratio_range,
                        tol=cfg.probes.geometric.tol,
                    )
                    rec.dropout_star = d_star

                # interpolation (within class if possible: select another sample j with same label)
                if cfg.probes.interpolation.enabled:
                    # naive: use next sample in batch with same label if any
                    alpha_star = None
                    for j in range(x.size(0)):
                        if j == i:
                            continue
                        if int(y[j].item()) == int(yi.item()):
                            alpha_star = search_interpolation_boundary(
                                model, x[i].detach().cpu(), x[j].detach().cpu(), yi, steps=cfg.probes.interpolation.steps
                            )
                            break
                    rec.alpha_star = alpha_star

                sample_records.append(rec)
                sample_id += 1
            if cfg.general.sample_limit is not None and sample_id >= cfg.general.sample_limit:
                break

        # write partial outputs
        write_csv([r.as_dict() for r in sample_records], os.path.join(self.out_dir, "metrics.csv"))
        write_csv([r.as_dict() for r in layerwise_records], os.path.join(self.out_dir, "layerwise_topology.csv"))
        write_csv([r.as_dict() for r in diagdist_records], os.path.join(self.out_dir, "diagram_distances.csv"))

        # curves
        for norm, ys in ra_curves.items():
            xs = self.cfg.probes.adversarial.eps_grid
            save_curve_png(xs, ys, f"RA vs epsilon ({norm})", os.path.join(self.out_dir, f"ra_curve_{norm}.png"))

        return {
            "sample_records": sample_records,
            "layerwise_records": layerwise_records,
            "diagdist_records": diagdist_records,
            "ra_curves": ra_curves,
        }

    def aggregate_and_report(self, results):
        recs: List[SampleRecord] = results["sample_records"]
        ra_curves = results["ra_curves"]
        diagdist_records: List[DiagramDistanceRecord] = results["diagdist_records"]

        def mean_safe(vals):
            arr = [v for v in vals if v is not None and not (isinstance(v, float) and (np.isnan(v)))]
            return float(np.mean(arr)) if arr else None

        summary = RobustnessSummary()
        summary.ra_curves = ra_curves
        summary.mean_eps_star_linf = mean_safe([r.eps_star_linf for r in recs])
        summary.mean_eps_star_l2 = mean_safe([r.eps_star_l2 for r in recs])

        # plots
        if self.cfg.reporting.save_plots:
            save_hist_png([r.eps_star_linf for r in recs], "eps* L_inf", os.path.join(self.out_dir, "hist_eps_linf.png"))
            save_hist_png([r.eps_star_l2 for r in recs], "eps* L2", os.path.join(self.out_dir, "hist_eps_l2.png"))
            # simple per-layer average Wasserstein H0/H1 bars
            from collections import defaultdict
            from .report import (
                save_distance_heatmap,
                save_distance_heatmap_normalized,
                save_scatter_eps_vs_distance,
                save_distance_curves_with_ci,
                save_violin_distance_by_layer,
                best_signal_layerH,
            )
            for H in [0, 1]:
                agg = defaultdict(list)
                for r in diagdist_records:
                    if r.metric == "wasserstein" and r.H == H and not (r.distance is None or (isinstance(r.distance, float) and np.isnan(r.distance))):
                        agg[r.layer].append(r.distance)
                avg = {layer: float(np.mean(vals)) for layer, vals in agg.items() if vals}
                if avg:
                    save_layer_distance_bar(avg, f"Wasserstein H{H} per layer", os.path.join(self.out_dir, f"layer_wasserstein_H{H}.png"))
            # Betti count bars (clean stats averaged, stored in layerwise_records)
            from .report import save_betti_counts_bar as _save_betti
            _save_betti(results["layerwise_records"], 0, os.path.join(self.out_dir, "betti_H0_counts.png"))
            _save_betti(results["layerwise_records"], 1, os.path.join(self.out_dir, "betti_H1_counts.png"))
            # Heatmaps (raw and normalized by noise floor)
            save_distance_heatmap(diagdist_records, "wasserstein", 1, os.path.join(self.out_dir, "heatmap_wasserstein_H1.png"), "Wasserstein H1 (mean)")
            save_distance_heatmap(diagdist_records, "wasserstein", 0, os.path.join(self.out_dir, "heatmap_wasserstein_H0.png"), "Wasserstein H0 (mean)")
            save_distance_heatmap_normalized(diagdist_records, "wasserstein", 1, os.path.join(self.out_dir, "heatmap_wasserstein_H1_norm.png"), "Wasserstein H1 (minus noise floor)")
            save_distance_heatmap_normalized(diagdist_records, "wasserstein", 0, os.path.join(self.out_dir, "heatmap_wasserstein_H0_norm.png"), "Wasserstein H0 (minus noise floor)")
            # Distance vs epsilon curves (top-k layers)
            save_distance_curves_with_ci(diagdist_records, "wasserstein", 1, "linf", os.path.join(self.out_dir, "curves_wasserstein_H1_linf.png"))
            save_distance_curves_with_ci(diagdist_records, "wasserstein", 1, "l2", os.path.join(self.out_dir, "curves_wasserstein_H1_l2.png"))
            # Pick best signal layer/H for scatter
            best = best_signal_layerH(diagdist_records, metric="wasserstein")
            if best is not None:
                best_layer, best_H = best
                save_scatter_eps_vs_distance(recs, diagdist_records, "linf", best_layer, "wasserstein", best_H, None, os.path.join(self.out_dir, f"scatter_eps_linf_vs_dist_H{best_H}_{best_layer}.png"))
                save_scatter_eps_vs_distance(recs, diagdist_records, "l2", best_layer, "wasserstein", best_H, None, os.path.join(self.out_dir, f"scatter_eps_l2_vs_dist_H{best_H}_{best_layer}.png"))
            # Violin distributions at max eps
            save_violin_distance_by_layer(diagdist_records, "wasserstein", 1, "linf", "max", os.path.join(self.out_dir, "violin_wasserstein_H1_linf.png"))
            save_violin_distance_by_layer(diagdist_records, "wasserstein", 1, "l2", "max", os.path.join(self.out_dir, "violin_wasserstein_H1_l2.png"))

        # write summary.json
        summary.to_json(os.path.join(self.out_dir, "summary.json"))


