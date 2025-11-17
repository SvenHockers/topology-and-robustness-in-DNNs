from __future__ import annotations

import os
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

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
        logging.info(f"Pipeline initialized (device={self.device}, exp='{self.cfg.general.exp_name}')")

    def prepare_data(self):
        d = self.cfg.data
        logging.info(f"Preparing data (n_samples_per_shape={d.n_samples_per_shape}, n_points={d.n_points}, noise={d.noise})")
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
        logging.info(f"Data prepared: train={len(train_ds)}, val={len(val_ds)}, batch_size={d.batch_size}")
        # optional: visualize sample diagrams per class
        # This is wrapped in try-except to ensure pipeline continues even if visualization fails
        if self.cfg.reporting.save_plots and self.cfg.reporting.sample_visualizations_per_class:
            try:
                from ..visualization import visualize_sample_diagrams
                save_path = os.path.join(self.out_dir, "persistence_diagrams_by_class.png")
                result = visualize_sample_diagrams(
                    self._raw_point_clouds,
                    self._raw_labels,
                    n_samples_per_class=int(self.cfg.reporting.sample_visualizations_per_class),
                    maxdim=int(self.cfg.probes.topology.maxdim) if hasattr(self.cfg.probes, "topology") else 2,
                    save_path=save_path,
                    show=False,
                    seed=int(self.cfg.general.seed),
                )
                if result is not None and os.path.exists(save_path):
                    logging.info(f"Wrote {save_path}")
                else:
                    logging.info(f"Skipped persistence_diagrams_by_class (non-critical)")
            except Exception as _e:
                # Ensure pipeline continues even if visualization completely fails
                logging.warning(f"Visualization failed (non-critical, continuing): {type(_e).__name__}")
                # Don't re-raise - let pipeline continue

    def prepare_model(self):
        m = self.cfg.model
        logging.info(f"Preparing model (arch={m.arch}, train={m.train}, epochs={m.epochs}, lr={m.lr})")
        if m.arch == "MLP":
            model = SimplePointMLP(num_classes=3)
        else:
            model = SimplePointCNN(num_classes=3)
        self._model = model.to(self.device)

        if m.checkpoint and os.path.exists(m.checkpoint):
            state = torch.load(m.checkpoint, map_location=self.device)
            self._model.load_state_dict(state)
            self._model.eval()
            logging.info(f"Loaded checkpoint: {m.checkpoint}")
            return

        if m.train:
            optimizer = torch.optim.Adam(self._model.parameters(), lr=m.lr)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(1, m.epochs + 1):
                train_one_epoch(self._model, self._train_loader, optimizer, criterion, self.device)
            # quick val to warm-up
            v_loss, v_acc = evaluate(self._model, self._val_loader, criterion, self.device)
            logging.info(f"Post-train eval: val_loss={v_loss:.4f}, val_acc={v_acc:.3f}")
            # save checkpoint
            torch.save(self._model.state_dict(), os.path.join(self.out_dir, "model.pth"))
            logging.info(f"Saved checkpoint to {os.path.join(self.out_dir, 'model.pth')}")
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
        # Store adversarial examples and original samples for visualization
        adversarial_examples: Dict[int, Dict[str, torch.Tensor]] = {}  # sample_id -> {norm: x_adv}
        original_samples: Dict[int, torch.Tensor] = {}  # sample_id -> x_clean
        # Store samples for layer transformation visualization (always store a few)
        layer_viz_samples: Dict[int, torch.Tensor] = {}  # sample_id -> x_clean

        # robust accuracy curves
        ra_curves: Dict[str, List[float]] = {}
        if cfg.probes.adversarial.enabled and cfg.probes.adversarial.eps_grid:
            logging.info(f"Computing RA curves over eps_grid={cfg.probes.adversarial.eps_grid}")
            for norm in cfg.probes.adversarial.norms:
                ra = robust_accuracy_curve(model, self._val_loader, norm, cfg.probes.adversarial.eps_grid, cfg.probes.adversarial.steps)
                ra_curves[norm] = ra
                logging.info(f"RA({norm})={ra}")

        # per-sample probes
        model.eval()
        sample_id = 0
        logging.info("Running per-sample probes...")
        for batch_idx, (x, y) in enumerate(self._val_loader):
            x, y = x.to(device), y.to(device)
            for i in range(x.size(0)):
                if cfg.general.sample_limit is not None and sample_id >= cfg.general.sample_limit:
                    logging.info(f"Reached sample_limit={cfg.general.sample_limit}")
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
                # Store original sample for visualization
                if cfg.reporting.save_adversarial_visualizations:
                    original_samples[sample_id] = xi
                # Store samples for layer transformation visualization (store diverse samples)
                if cfg.reporting.save_layer_transformations:
                    # Store one sample per class, up to n_layer_transformation_samples
                    from collections import defaultdict
                    if not hasattr(cfg.reporting, 'n_layer_transformation_samples'):
                        n_target = 3
                    else:
                        n_target = cfg.reporting.n_layer_transformation_samples
                    class_counts = defaultdict(int)
                    for sid in layer_viz_samples.keys():
                        rec_existing = next((r for r in sample_records if r.sample_id == sid), None)
                        if rec_existing:
                            class_counts[rec_existing.true_label] += 1
                    current_class_count = class_counts.get(int(yi.item()), 0)
                    n_per_class = max(1, n_target // 3)  # Assuming 3 classes
                    if current_class_count < n_per_class and len(layer_viz_samples) < n_target:
                        layer_viz_samples[sample_id] = xi

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
                                # Store for visualization
                                if cfg.reporting.save_adversarial_visualizations:
                                    if sample_id not in adversarial_examples:
                                        adversarial_examples[sample_id] = {}
                                    adversarial_examples[sample_id][norm] = x_adv
                        else:
                            rec.eps_star_l2 = eps_star
                            if x_adv is not None:
                                with torch.no_grad():
                                    pred_adv = int(model(x_adv.to(device), save_layers=False).argmax(1).item())
                                rec.adv_pred_l2 = pred_adv
                                # Store for visualization
                                if cfg.reporting.save_adversarial_visualizations:
                                    if sample_id not in adversarial_examples:
                                        adversarial_examples[sample_id] = {}
                                    adversarial_examples[sample_id][norm] = x_adv

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
                if sample_id % 20 == 0:
                    logging.info(f"Processed {sample_id} samples")
            if cfg.general.sample_limit is not None and sample_id >= cfg.general.sample_limit:
                break

        # write partial outputs
        write_csv([r.as_dict() for r in sample_records], os.path.join(self.out_dir, "metrics.csv"))
        write_csv([r.as_dict() for r in layerwise_records], os.path.join(self.out_dir, "layerwise_topology.csv"))
        write_csv([r.as_dict() for r in diagdist_records], os.path.join(self.out_dir, "diagram_distances.csv"))
        logging.info(f"Wrote CSVs (metrics={len(sample_records)}, layerwise_topology={len(layerwise_records)}, diagram_distances={len(diagdist_records)})")

        # curves
        for norm, ys in ra_curves.items():
            xs = self.cfg.probes.adversarial.eps_grid
            save_curve_png(xs, ys, f"RA vs epsilon ({norm})", os.path.join(self.out_dir, f"ra_curve_{norm}.png"))
            logging.info(f"Wrote RA curve plot for norm={norm}")

        return {
            "sample_records": sample_records,
            "layerwise_records": layerwise_records,
            "diagdist_records": diagdist_records,
            "ra_curves": ra_curves,
            "adversarial_examples": adversarial_examples,
            "original_samples": original_samples,
            "layer_viz_samples": layer_viz_samples,
        }

    def aggregate_and_report(self, results):
        recs: List[SampleRecord] = results["sample_records"]
        ra_curves = results["ra_curves"]
        diagdist_records: List[DiagramDistanceRecord] = results["diagdist_records"]
        adversarial_examples = results.get("adversarial_examples", {})
        original_samples = results.get("original_samples", {})
        layer_viz_samples = results.get("layer_viz_samples", {})

        def mean_safe(vals):
            arr = [v for v in vals if v is not None and not (isinstance(v, float) and (np.isnan(v)))]
            return float(np.mean(arr)) if arr else None

        summary = RobustnessSummary()
        summary.ra_curves = ra_curves
        summary.mean_eps_star_linf = mean_safe([r.eps_star_linf for r in recs])
        summary.mean_eps_star_l2 = mean_safe([r.eps_star_l2 for r in recs])
        logging.info(f"Aggregates: mean_eps_star_linf={summary.mean_eps_star_linf}, mean_eps_star_l2={summary.mean_eps_star_l2}")

        # plots
        if self.cfg.reporting.save_plots:
            save_hist_png([r.eps_star_linf for r in recs], "eps* L_inf", os.path.join(self.out_dir, "hist_eps_linf.png"))
            save_hist_png([r.eps_star_l2 for r in recs], "eps* L2", os.path.join(self.out_dir, "hist_eps_l2.png"))
            logging.info("Wrote eps* histograms")
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
                    save_layer_distance_bar(avg, f"Wasserstein $H_{{{H}}}$ per layer", os.path.join(self.out_dir, f"layer_wasserstein_H{H}.png"))
            # Betti count bars (clean stats averaged, stored in layerwise_records)
            from .report import save_betti_counts_bar as _save_betti
            _save_betti(results["layerwise_records"], 0, os.path.join(self.out_dir, "betti_H0_counts.png"))
            _save_betti(results["layerwise_records"], 1, os.path.join(self.out_dir, "betti_H1_counts.png"))
            logging.info("Wrote Betti count bars")
            # Heatmaps (raw and normalized by noise floor)
            save_distance_heatmap(diagdist_records, "wasserstein", 1, os.path.join(self.out_dir, "heatmap_wasserstein_H1.png"), "Wasserstein $H_{1}$ (mean)")
            save_distance_heatmap(diagdist_records, "wasserstein", 0, os.path.join(self.out_dir, "heatmap_wasserstein_H0.png"), "Wasserstein $H_{0}$ (mean)")
            save_distance_heatmap_normalized(diagdist_records, "wasserstein", 1, os.path.join(self.out_dir, "heatmap_wasserstein_H1_norm.png"), "Wasserstein $H_{1}$ (minus noise floor)")
            save_distance_heatmap_normalized(diagdist_records, "wasserstein", 0, os.path.join(self.out_dir, "heatmap_wasserstein_H0_norm.png"), "Wasserstein $H_{0}$ (minus noise floor)")
            logging.info("Wrote heatmaps (raw & normalized)")
            # Distance vs epsilon curves (top-k layers)
            save_distance_curves_with_ci(diagdist_records, "wasserstein", 1, "linf", os.path.join(self.out_dir, "curves_wasserstein_H1_linf.png"))
            save_distance_curves_with_ci(diagdist_records, "wasserstein", 1, "l2", os.path.join(self.out_dir, "curves_wasserstein_H1_l2.png"))
            logging.info("Wrote distance-vs-epsilon curves")
            # Pick best signal layer/H for scatter
            best = best_signal_layerH(diagdist_records, metric="wasserstein")
            if best is not None:
                best_layer, best_H = best
                save_scatter_eps_vs_distance(recs, diagdist_records, "linf", best_layer, "wasserstein", best_H, None, os.path.join(self.out_dir, f"scatter_eps_linf_vs_dist_H{best_H}_{best_layer}.png"))
                save_scatter_eps_vs_distance(recs, diagdist_records, "l2", best_layer, "wasserstein", best_H, None, os.path.join(self.out_dir, f"scatter_eps_l2_vs_dist_H{best_H}_{best_layer}.png"))
                logging.info(f"Wrote scatter plots (best layer={best_layer}, H={best_H})")
            # Violin distributions at max eps
            save_violin_distance_by_layer(diagdist_records, "wasserstein", 1, "linf", "max", os.path.join(self.out_dir, "violin_wasserstein_H1_linf.png"))
            save_violin_distance_by_layer(diagdist_records, "wasserstein", 1, "l2", "max", os.path.join(self.out_dir, "violin_wasserstein_H1_l2.png"))
            logging.info("Wrote violin plots")
            
            # Create subdirectories for new visualizations
            viz_dir = os.path.join(self.out_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            adv_viz_dir = os.path.join(viz_dir, "adversarial_examples")
            per_class_dir = os.path.join(viz_dir, "per_class")
            stat_dir = os.path.join(viz_dir, "statistical")
            layer_viz_dir = os.path.join(viz_dir, "layer_transformations")
            os.makedirs(adv_viz_dir, exist_ok=True)
            os.makedirs(per_class_dir, exist_ok=True)
            os.makedirs(stat_dir, exist_ok=True)
            os.makedirs(layer_viz_dir, exist_ok=True)
            
            # Adversarial example visualizations
            if self.cfg.reporting.save_adversarial_visualizations and adversarial_examples:
                from .report import save_adversarial_example, save_sample_grid
                # Save individual examples
                n_viz = min(self.cfg.reporting.n_adversarial_visualizations, len(adversarial_examples))
                selected_ids = list(adversarial_examples.keys())[:n_viz]
                for sample_id in selected_ids:
                    if sample_id not in original_samples:
                        continue
                    x_orig = original_samples[sample_id]
                    for norm in adversarial_examples[sample_id].keys():
                        x_adv = adversarial_examples[sample_id][norm]
                        save_path = os.path.join(adv_viz_dir, f"sample_{sample_id}_{norm}.png")
                        save_adversarial_example(x_orig, x_adv, save_path, title_suffix=f"Sample {sample_id} ({norm})")
                logging.info(f"Wrote {len(selected_ids)} adversarial example visualizations")
                # Save sample grid
                save_sample_grid(recs, original_samples, adversarial_examples, adv_viz_dir, 
                               n_samples=min(9, len(adversarial_examples)),
                               selection=self.cfg.reporting.visualization_selection)
                logging.info("Wrote sample grid")
            
            # Per-class visualizations
            if self.cfg.reporting.save_per_class_plots:
                from .report import (
                    save_eps_star_by_class_boxplot,
                    save_class_confusion_matrix,
                    save_topology_distance_by_class,
                    save_robust_accuracy_by_class,
                )
                # Eps* by class
                for norm in self.cfg.probes.adversarial.norms:
                    save_eps_star_by_class_boxplot(recs, norm, os.path.join(per_class_dir, f"eps_star_by_class_{norm}.png"))
                    save_class_confusion_matrix(recs, norm, os.path.join(per_class_dir, f"confusion_matrix_{norm}.png"))
                logging.info("Wrote per-class eps* and confusion matrices")
                # Robust accuracy by class (if per-class data available)
                if self.cfg.probes.adversarial.enabled and self.cfg.probes.adversarial.eps_grid:
                    # Recompute with per_class=True
                    ra_curves_by_class = {}
                    for norm in self.cfg.probes.adversarial.norms:
                        from .probes import robust_accuracy_curve
                        ra_by_class = robust_accuracy_curve(
                            self._model, self._val_loader, norm, 
                            self.cfg.probes.adversarial.eps_grid, 
                            self.cfg.probes.adversarial.steps,
                            per_class=True
                        )
                        if isinstance(ra_by_class, dict):
                            ra_curves_by_class[norm] = ra_by_class
                            save_robust_accuracy_by_class(
                                ra_by_class, self.cfg.probes.adversarial.eps_grid, norm,
                                os.path.join(per_class_dir, f"robust_accuracy_by_class_{norm}.png")
                            )
                    logging.info("Wrote robust accuracy by class")
                # Topology distance by class (for a few key conditions)
                if diagdist_records:
                    # Find a representative layer and condition
                    from collections import defaultdict
                    condition_counts = defaultdict(int)
                    for r in diagdist_records:
                        if r.condition.startswith("adv_"):
                            condition_counts[r.condition] += 1
                    if condition_counts:
                        # Use most common condition
                        common_condition = max(condition_counts.items(), key=lambda x: x[1])[0]
                        # Find a layer with data
                        layer_counts = defaultdict(int)
                        for r in diagdist_records:
                            if r.condition == common_condition:
                                layer_counts[r.layer] += 1
                        if layer_counts:
                            common_layer = max(layer_counts.items(), key=lambda x: x[1])[0]
                            save_topology_distance_by_class(
                                diagdist_records, recs, "wasserstein", 1, common_layer, common_condition,
                                os.path.join(per_class_dir, f"topology_distance_by_class_{common_layer}_{common_condition}.png")
                            )
                            logging.info("Wrote topology distance by class")
            
            # Statistical plots
            if self.cfg.reporting.save_statistical_plots:
                from .report import (
                    save_correlation_heatmap,
                    save_percentile_curves,
                    save_outlier_analysis,
                )
                # Correlation heatmap
                save_correlation_heatmap(recs, diagdist_records, os.path.join(stat_dir, "correlation_heatmap.png"))
                logging.info("Wrote correlation heatmap")
                # Percentile curves (for best layer)
                best = best_signal_layerH(diagdist_records, metric="wasserstein")
                if best is not None:
                    best_layer, best_H = best
                    for norm in self.cfg.probes.adversarial.norms:
                        save_percentile_curves(
                            diagdist_records, "wasserstein", best_H, norm, best_layer,
                            os.path.join(stat_dir, f"percentile_curves_{norm}_{best_layer}_H{best_H}.png")
                        )
                    logging.info("Wrote percentile curves")
                # Outlier analysis
                for norm in self.cfg.probes.adversarial.norms:
                    save_outlier_analysis(recs, norm, os.path.join(stat_dir, f"outlier_analysis_{norm}.png"))
                logging.info("Wrote outlier analysis")
            
            # Intuitive comparison visualizations
            if self.cfg.reporting.save_plots:
                intuitive_dir = os.path.join(viz_dir, "intuitive")
                os.makedirs(intuitive_dir, exist_ok=True)
                
                from .report import (
                    save_norm_comparison_bars,
                    save_robustness_summary_cards,
                    save_class_robustness_comparison,
                    save_layer_sensitivity_comparison,
                    save_robustness_radar,
                    save_epsilon_impact_comparison,
                )
                
                # Norm comparison (L∞ vs L2)
                save_norm_comparison_bars(recs, os.path.join(intuitive_dir, "norm_comparison.png"))
                logging.info("Wrote norm comparison chart")
                
                # Summary cards (4-panel overview)
                if self.cfg.probes.adversarial.eps_grid:
                    save_robustness_summary_cards(
                        recs, ra_curves, self.cfg.probes.adversarial.eps_grid,
                        os.path.join(intuitive_dir, "summary_cards.png")
                    )
                    logging.info("Wrote robustness summary cards")
                
                # Class robustness comparison
                save_class_robustness_comparison(recs, os.path.join(intuitive_dir, "class_robustness_comparison.png"))
                logging.info("Wrote class robustness comparison")
                
                # Layer sensitivity comparison
                best = best_signal_layerH(diagdist_records, metric="wasserstein")
                if best is not None:
                    best_layer, best_H = best
                    for norm in self.cfg.probes.adversarial.norms:
                        save_layer_sensitivity_comparison(
                            diagdist_records, "wasserstein", best_H, norm,
                            os.path.join(intuitive_dir, f"layer_sensitivity_{norm}_H{best_H}.png")
                        )
                    logging.info("Wrote layer sensitivity comparisons")
                
                # Epsilon impact comparison
                for norm in self.cfg.probes.adversarial.norms:
                    save_epsilon_impact_comparison(
                        diagdist_records, "wasserstein", 1, norm,
                        os.path.join(intuitive_dir, f"epsilon_impact_{norm}.png")
                    )
                logging.info("Wrote epsilon impact comparisons")
                
                # Radar chart (multi-dimensional profile)
                save_robustness_radar(recs, diagdist_records, os.path.join(intuitive_dir, "robustness_radar.png"))
                logging.info("Wrote robustness radar chart")
            
            # Layer transformation visualizations
            if self.cfg.reporting.save_layer_transformations:
                from .report import (
                    save_layer_transformation_grid,
                    save_layer_transformation_comparison,
                )
                # Select diverse samples (one per class if possible)
                from collections import defaultdict
                samples_by_class = defaultdict(list)
                for r in recs:
                    samples_by_class[r.true_label].append(r.sample_id)
                
                selected_samples = []
                n_per_class = max(1, self.cfg.reporting.n_layer_transformation_samples // len(samples_by_class) if samples_by_class else 1)
                for class_id, sample_ids in samples_by_class.items():
                    selected_samples.extend(sample_ids[:n_per_class])
                selected_samples = selected_samples[:self.cfg.reporting.n_layer_transformation_samples]
                
                # Get layers to visualize
                layers = None
                if self.cfg.probes.layerwise_topology.enabled:
                    layers = self.cfg.probes.layerwise_topology.layers
                
                # Visualize layer transformations for selected samples
                # Use stored layer_viz_samples if available, otherwise fall back to original_samples
                samples_to_use = layer_viz_samples if layer_viz_samples else original_samples
                
                for sample_id in selected_samples:
                    # Find the sample record
                    sample_rec = next((r for r in recs if r.sample_id == sample_id), None)
                    if sample_rec is None:
                        continue
                    
                    # Get the original sample
                    if sample_id in samples_to_use:
                        x_sample = samples_to_use[sample_id]
                    elif sample_id in original_samples:
                        x_sample = original_samples[sample_id]
                    else:
                        # Fallback: try to find in validation loader (less reliable)
                        found = False
                        for x_batch, y_batch in self._val_loader:
                            for i in range(x_batch.size(0)):
                                if int(y_batch[i].item()) == sample_rec.true_label:
                                    x_sample = x_batch[i].detach().cpu()
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            continue
                    
                    # Get reduction settings from topology config (for consistency)
                    reduction_method = "pca"  # Default, could be made configurable
                    normalize = self.cfg.probes.topology.normalize if self.cfg.probes.topology.enabled else "zscore"
                    pca_dim = self.cfg.probes.topology.pca_dim if self.cfg.probes.topology.enabled else None
                    
                    # Visualize clean transformations
                    save_path = os.path.join(layer_viz_dir, f"layer_transformations_sample_{sample_id}.png")
                    try:
                        save_layer_transformation_grid(
                            self._model,
                            x_sample,
                            sample_id,
                            sample_rec.true_label,
                            save_path,
                            layers=layers,
                            reduction_method=reduction_method,
                            normalize=normalize,
                            pca_dim=pca_dim,
                        )
                        logging.info(f"Wrote layer transformation grid for sample {sample_id}")
                    except Exception as e:
                        logging.warning(f"Failed to create layer transformation for sample {sample_id}: {e}")
                    
                    # If adversarial examples are available, create comparison
                    if sample_id in adversarial_examples and sample_id in original_samples:
                        for norm in adversarial_examples[sample_id].keys():
                            x_clean = original_samples[sample_id]
                            x_adv = adversarial_examples[sample_id][norm]
                            save_path = os.path.join(layer_viz_dir, f"layer_transformations_clean_vs_adv_sample_{sample_id}_{norm}.png")
                            try:
                                save_layer_transformation_comparison(
                                    self._model,
                                    x_clean,
                                    x_adv,
                                    sample_id,
                                    sample_rec.true_label,
                                    save_path,
                                    layers=layers,
                                    norm=norm,
                                    reduction_method=reduction_method,
                                    normalize=normalize,
                                    pca_dim=pca_dim,
                                )
                                logging.info(f"Wrote layer transformation comparison for sample {sample_id} ({norm})")
                            except Exception as e:
                                logging.warning(f"Failed to create layer comparison for sample {sample_id} ({norm}): {e}")

        # write summary.json
        summary.to_json(os.path.join(self.out_dir, "summary.json"))
        logging.info("Wrote summary.json")


