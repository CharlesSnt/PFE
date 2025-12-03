import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import os

# Import Definition
from NavierStokespytorch import PhysicsInformedCNN, pinn_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    print(f"\n--- Traitement de : {title} ---")

    model = PhysicsInformedCNN(in_channels=3, hidden_channels=32, out_channels=3).to(
        device
    )
    if not os.path.exists(model_path) or not os.path.exists(traj_path):
        print(f"Fichiers manquants pour {title} ({model_path}). Passage.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    w_final = parameters_to_vector(model.parameters()).detach()
    history = torch.load(traj_path, map_location="cpu")  # Trajectoire sur CPU

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

    print("Loss Landscape calculus...")
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            vec = w_final + a * dir_x + b * dir_y
            vector_to_parameters(vec, model.parameters())
            # IMPORTANT: No torch.no_grad() for the PINNs
            l = pinn_loss(model, xyt_val, u_val, v_val).item()
            loss_grid[i, j] = np.log10(l + 1e-10)

    vector_to_parameters(w_final, model.parameters())

    # Plot
    X, Y = np.meshgrid(alphas, betas)
    plt.figure(figsize=(10, 8))
    cp = plt.contourf(X, Y, loss_grid.T, levels=30, cmap="Spectral_r")
    plt.colorbar(cp, label="Log10(Loss)")

    plt.plot(
        traj_x,
        traj_y,
        color="white",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Trajectory",
    )
    plt.scatter(traj_x[0], traj_y[0], c="white", marker="o", s=50, label="Start (Init)")
    plt.scatter(0, 0, c="black", marker="*", s=200, label="End (Min)")

    plt.title(f"Loss Landscape : {title}")
    plt.xlabel("Principal Direction (Init -> Min)")
    plt.ylabel("Orthogonal Direction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Sauvegardé : {output_filename}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    data = scipy.io.loadmat("Data/cylinder_nektar_wake.mat")
    U_star, t_star, X_star = data["U_star"], data["t"], data["X_star"]

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

    # --- Generating plots ---

    # 1. Adam + LBFGS
    generate_landscape_plot(
        "navier_adam_lbfgs_model.pth",
        "navier_adam_lbfgs_traj.pt",
        "Adam + LBFGS",
        "landscape_adam_lbfgs.png",
        xyt_val,
        u_val,
        v_val,
    )

    # 2. Adam Only
    generate_landscape_plot(
        "navier_adam_only_model.pth",
        "navier_adam_only_traj.pt",
        "Adam (Without LBFGS)",
        "landscape_adam_only.png",
        xyt_val,
        u_val,
        v_val,
    )

    # 3. SGD
    generate_landscape_plot(
        "navier_sgd_model.pth",
        "navier_sgd_traj.pt",
        "Stochastic Gradient Descent",
        "landscape_sgd.png",
        xyt_val,
        u_val,
        v_val,
    )
