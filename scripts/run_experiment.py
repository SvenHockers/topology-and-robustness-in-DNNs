import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import GiottoPointCloudDataset, make_point_clouds
from src.models import SimplePointCNN, SimplePointMLP
from src.topology import analyze_layer_topology, visualize_layer_topology
from src.attacks import (
    find_min_adversarial_perturbation_iterative,
    find_one_correct_sample_of_class,
)
from src.training import show_some_predictions, train_one_epoch, evaluate
from src.visualization import (
    visualize_sample_diagrams,
    plot_original_vs_adversarial,
    plot_torus_wireframe_compare,
)
from src.plot_style import use_exploratory_style


def main():
    # Exploratory style for local experimentation/visualization
    use_exploratory_style()
    print("Generating point clouds...")
    point_clouds, labels = make_point_clouds(n_samples_per_shape=50, n_points=20, noise=0.1)
    print(f"Generated {len(point_clouds)} point clouds")
    print(f"Shape: {point_clouds.shape}, Labels: {labels.shape}")

    # Visualize sample persistence diagrams BEFORE training
    print("\nVisualizing persistence diagrams for sample point clouds...")
    visualize_sample_diagrams(point_clouds, labels)

    # Split into train/val
    n_total = len(labels)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_ds = GiottoPointCloudDataset(point_clouds[train_idx], labels[train_idx])
    val_ds = GiottoPointCloudDataset(point_clouds[val_idx], labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Train both models and analyze topology
    for ModelClass, name in [(SimplePointMLP, "MLP"), (SimplePointCNN, "CNN")]:
        print(f"\n{'='*60}")
        print(f"Training and Analyzing {name} Model")
        print(f"{'='*60}")

        model = ModelClass(num_classes=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(1, 21):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

        # --- 1. Show some predictions (to see classification) ---
        print("\nSome validation predictions (true, pred):")
        show_some_predictions(model, val_loader, device, n_show=10)

        # --- 2. Torus-specific decision boundary / adversarial analysis (ITERATIVE) ---
        print(f"\nFinding minimal perturbation for a TORUS sample ({name} model) with iterative attack...")

        # Class 2 = torus (as defined in make_point_clouds)
        x_torus, y_torus = find_one_correct_sample_of_class(model, val_loader, device, class_id=2)

        if x_torus is None:
            print("Could not find a correctly classified torus sample in validation set.")
        else:
            eps_max = 1.0   # maximum allowed L_inf radius
            n_steps = 40    # number of gradient steps

            eps_star, x_adv, clean_pred, adv_pred = find_min_adversarial_perturbation_iterative(
                model, x_torus, y_torus, device, eps_max=eps_max, n_steps=n_steps
            )

            if eps_star is None:
                print(f"No misclassification within eps_max = {eps_max}.")
            else:
                print(f"Original label          : {int(y_torus.item())} (torus)")
                print(f"Clean prediction        : {clean_pred}")
                print(f"Adversarial prediction  : {adv_pred}")
                print(f"Minimal epsilon (L_inf) along this path: {eps_star:.4f}")

                # visualize geometry difference: points + displacement
                plot_original_vs_adversarial(x_torus, x_adv, title_suffix=f"(Torus, {name})")

                # and as a wireframe torus (original vs adversarial)
                plot_torus_wireframe_compare(x_torus, x_adv, title_suffix=f"(Torus, {name})")

        # Topology analysis after training
        print(f"\nAnalyzing layer topology for {name}...")
        topology_stats = analyze_layer_topology(model, val_loader, device)

        print(f"\nTopology Statistics per Layer:")
        for layer_name, stats in topology_stats.items():
            print(f"\n{layer_name}:")
            for stat_name, value in sorted(stats.items()):
                print(f"  {stat_name}: {value:.4f}")

        # Visualize persistence diagrams
        print(f"\nGenerating topology visualizations for {name}...")
        visualize_layer_topology(model, val_loader, device, name)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()


