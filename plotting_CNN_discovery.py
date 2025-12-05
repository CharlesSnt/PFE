import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import os

from CNN_discovery import PhysicsInformedCNN, pinn_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Utilitary functions
# -----------------------------
def get_projection_directions(w_final, w_init):
    vec_main = w_init - w_final
    norm_main = torch.norm(vec_main)
    if norm_main < 1e-6:
        dir_x = torch.randn_like(w_final)
    else:
        dir_x = vec_main / norm_main

    random_vec = torch.randn_like(w_final)
    dir_y = random_vec - torch.dot(random_vec, dir_x) * dir_x
    dir_y = dir_y / (torch.norm(dir_y) + 1e-10)
    return dir_x, dir_y


def project_trajectory(history, w_final, dir_x, dir_y):
    traj_x, traj_y = [], []
    w_final_cpu = w_final.cpu()
    dir_x_cpu, dir_y_cpu = dir_x.cpu(), dir_y.cpu()

    for w in history:
        diff = w - w_final_cpu
        traj_x.append(torch.dot(diff, dir_x_cpu).item())
        traj_y.append(torch.dot(diff, dir_y_cpu).item())
    return traj_x, traj_y


def generate_landscape_plot(
    model_path, traj_path, title, output_filename, xyt_val, u_val, v_val
):
    """Generating the loss landscape"""
    print(f"\n--- Traitement Landscape : {title} ---")

    model = PhysicsInformedCNN(in_channels=3, hidden_channels=32, out_channels=3).to(
        device
    )
    if not os.path.exists(model_path) or not os.path.exists(traj_path):
        print(f"Missing file for {title}.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    w_final = parameters_to_vector(model.parameters()).detach()
    history = torch.load(traj_path, map_location="cpu")

    w_init = history[0].to(device)
    dir_x, dir_y = get_projection_directions(w_final, w_init)
    traj_x, traj_y = project_trajectory(history, w_final, dir_x, dir_y)

    x_min, x_max = min(traj_x), max(traj_x)
    y_min, y_max = min(traj_y), max(traj_y)
    span_x = max((x_max - x_min), 0.1) * 0.2
    span_y = max((y_max - y_min), 0.1) * 0.5

    res = 25
    alphas = np.linspace(x_min - span_x, x_max + span_x, res)
    betas = np.linspace(-span_y - 0.1, span_y + 0.1, res)
    loss_grid = np.zeros((res, res))

    dir_x, dir_y = dir_x.to(device), dir_y.to(device)

    print("Calcul de la Loss Landscape...")
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            vec = w_final + a * dir_x + b * dir_y
            vector_to_parameters(vec, model.parameters())
            l = pinn_loss(model, xyt_val, u_val, v_val).item()
            loss_grid[i, j] = np.log10(l + 1e-10)

    vector_to_parameters(w_final, model.parameters())

    X, Y = np.meshgrid(alphas, betas)
    plt.figure(figsize=(10, 8))
    cp = plt.contourf(X, Y, loss_grid.T, levels=30, cmap="Spectral_r")
    plt.colorbar(cp, label="Log10(Loss)")
    plt.plot(traj_x, traj_y, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    plt.scatter(traj_x[0], traj_y[0], c="white", marker="o", s=50, label="Début")
    plt.scatter(0, 0, c="black", marker="*", s=200, label="Fin (Min)")
    plt.title(f"Loss Landscape : {title}")
    plt.legend()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Sauvegardé : {output_filename}")


def plot_pressure_field(
    model_path, data, snap_idx=100, output_name="pressure_comparison_CNN_discovery.png"
):
    print(f"\n--- Generating the pressure plot ({output_name}) ---")

    if not os.path.exists(model_path):
        print(f"Modèle {model_path} introuvable.")
        return

    X_star = data["X_star"]  # Spatial coordinates (N, 2)
    t_star = data["t"]  # Time (T, 1)
    P_star = data["p_star"]  # Exact pressure (N, T)

    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]
    t_fixed = t_star[snap_idx, 0]

    xx = torch.tensor(x_star, dtype=torch.float32).to(device)
    yy = torch.tensor(y_star, dtype=torch.float32).to(device)
    tt = torch.tensor(np.full_like(x_star, t_fixed), dtype=torch.float32).to(device)

    xyt = torch.cat([xx, yy, tt], dim=1).unsqueeze(-1).unsqueeze(-1)

    model = PhysicsInformedCNN(in_channels=3, hidden_channels=32, out_channels=3).to(
        device
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        _, _, p_pred_tensor = model(xyt)
        p_pred = p_pred_tensor.squeeze().cpu().numpy()

    p_exact = P_star[:, snap_idx]

    lb = X_star.min(0)
    ub = X_star.max(0)
    nn_grid = 200
    x_space = np.linspace(lb[0], ub[0], nn_grid)
    y_space = np.linspace(lb[1], ub[1], nn_grid)
    X, Y = np.meshgrid(x_space, y_space)

    P_pred_grid = griddata(X_star, p_pred.flatten(), (X, Y), method="cubic")
    P_exact_grid = griddata(X_star, p_exact.flatten(), (X, Y), method="cubic")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # -- Predicted --
    h1 = ax[0].imshow(
        P_pred_grid,
        interpolation="nearest",
        cmap="rainbow",
        extent=[lb[0], ub[0], lb[1], ub[1]],
        origin="lower",
        aspect="auto",
    )
    fig.colorbar(h1, ax=ax[0])
    ax[0].set_title("Predicted pressure")
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("$y$")

    # -- Exact --
    h2 = ax[1].imshow(
        P_exact_grid,
        interpolation="nearest",
        cmap="rainbow",
        extent=[lb[0], ub[0], lb[1], ub[1]],
        origin="lower",
        aspect="auto",
    )
    fig.colorbar(h2, ax=ax[1])
    ax[1].set_title("Exact pressure")
    ax[1].set_xlabel("$x$")
    ax[1].set_ylabel("$y$")

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.show()
    print(f"Sauvegardé : {output_name}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("Loading the data...")
    try:
        data = scipy.io.loadmat("PFE/Data/cylinder_nektar_wake.mat")
    except FileNotFoundError:
        print("Erreur: Data/cylinder_nektar_wake.mat introuvable.")
        exit()

    U_star = data["U_star"]
    t_star = data["t"]
    X_star = data["X_star"]


    N_val = 2000
    idx = np.random.choice(X_star.shape[0] * t_star.shape[0], N_val, replace=False)

    XX = np.tile(X_star[:, 0:1], (1, t_star.shape[0])).flatten()
    YY = np.tile(X_star[:, 1:2], (1, t_star.shape[0])).flatten()
    TT = np.tile(t_star, (1, X_star.shape[0])).T.flatten()
    UU = U_star[:, 0, :].flatten()
    VV = U_star[:, 1, :].flatten()

    x_val = torch.tensor(XX[idx][:, None], dtype=torch.float32).to(device)
    y_val = torch.tensor(YY[idx][:, None], dtype=torch.float32).to(device)
    t_val = torch.tensor(TT[idx][:, None], dtype=torch.float32).to(device)
    u_val = torch.tensor(UU[idx][:, None], dtype=torch.float32).to(device)
    v_val = torch.tensor(VV[idx][:, None], dtype=torch.float32).to(device)

    xyt_val = torch.cat([x_val, y_val, t_val], dim=1).unsqueeze(-1).unsqueeze(-1)
    xyt_val.requires_grad_(True)
    u_val = u_val.unsqueeze(-1).unsqueeze(-1)
    v_val = v_val.unsqueeze(-1).unsqueeze(-1)

    # ---------------------------------------------------------
    # Generating landscapes
    # ---------------------------------------------------------
    generate_landscape_plot(
        "navier_adam_lbfgs_model.pth",
        "navier_adam_lbfgs_traj.pt",
        "Adam + LBFGS",
        "landscape_adam_lbfgs.png",
        xyt_val,
        u_val,
        v_val,
    )

    generate_landscape_plot(
        "navier_adam_only_model.pth",
        "navier_adam_only_traj.pt",
        "Adam Only",
        "landscape_adam_only.png",
        xyt_val,
        u_val,
        v_val,
    )

    generate_landscape_plot(
        "navier_sgd_model.pth",
        "navier_sgd_traj.pt",
        "SGD",
        "landscape_sgd.png",
        xyt_val,
        u_val,
        v_val,
    )

    # ---------------------------------------------------------
    # Pressure Graph
    # ---------------------------------------------------------

    plot_pressure_field(
        model_path="navier_adam_lbfgs_model.pth",
        data=data,
        snap_idx=100,
        output_name="pressure_comparison_adam_lbfgs_CNN_discovery.png",
    )
