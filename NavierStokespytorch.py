import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import numpy as np
from itertools import product, combinations

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Xavier initialization
# -----------------------------
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# -----------------------------
# CNN-based PINN
# -----------------------------
class PhysicsInformedCNN(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        # Xavier initialization
        self.apply(init_weights)

    def forward(self, xyt):
        h = torch.tanh(self.conv1(xyt))
        h = torch.tanh(self.conv2(h))
        h = torch.tanh(self.conv3(h))
        out = self.out_conv(h)
        u = out[:, 0:1, :, :]
        v = out[:, 1:2, :, :]
        p = out[:, 2:3, :, :]
        return u, v, p


# -----------------------------
# Gradient helper
# -----------------------------
def gradient(y, x):
    # Compute gradient
    grad = autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]

    # Replace None with zeros
    if grad is None:
        grad = torch.zeros_like(x)
    return grad


# -----------------------------
# Navier-Stokes residuals
# -----------------------------
def navier_stokes_residual(model, xyt, nu=0.01):
    xyt.requires_grad_(True)
    u, v, p = model(xyt)

    # gradients w.r.t the input channels
    u_x = autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True)[0][
        :, 0:1, :, :
    ]
    u_y = autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True)[0][
        :, 1:2, :, :
    ]
    u_t = autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True)[0][
        :, 2:3, :, :
    ]

    v_x = autograd.grad(v, xyt, grad_outputs=torch.ones_like(v), create_graph=True)[0][
        :, 0:1, :, :
    ]
    v_y = autograd.grad(v, xyt, grad_outputs=torch.ones_like(v), create_graph=True)[0][
        :, 1:2, :, :
    ]
    v_t = autograd.grad(v, xyt, grad_outputs=torch.ones_like(v), create_graph=True)[0][
        :, 2:3, :, :
    ]

    p_x = autograd.grad(p, xyt, grad_outputs=torch.ones_like(p), create_graph=True)[0][
        :, 0:1, :, :
    ]
    p_y = autograd.grad(p, xyt, grad_outputs=torch.ones_like(p), create_graph=True)[0][
        :, 1:2, :, :
    ]

    u_xx = autograd.grad(
        u_x, xyt, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0:1, :, :]
    u_yy = autograd.grad(
        u_y, xyt, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0][:, 1:2, :, :]

    v_xx = autograd.grad(
        v_x, xyt, grad_outputs=torch.ones_like(v_x), create_graph=True
    )[0][:, 0:1, :, :]
    v_yy = autograd.grad(
        v_y, xyt, grad_outputs=torch.ones_like(v_y), create_graph=True
    )[0][:, 1:2, :, :]

    f_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return f_u, f_v, u, v, p


# -----------------------------
# Loss function
# -----------------------------
def pinn_loss(model, xyt, u_true, v_true, nu=0.01):
    f_u, f_v, u_pred, v_pred, _ = navier_stokes_residual(model, xyt, nu)
    mse_data = ((u_pred - u_true) ** 2).mean() + ((v_pred - v_true) ** 2).mean()
    mse_pde = (f_u**2).mean() + (f_v**2).mean()
    return mse_data + mse_pde


# -----------------------------
# Training function
# -----------------------------
def train(model, xyt, u_true, v_true, adam_iters=5000, lbfgs_iters=500):
    model.to(device)
    xyt = xyt.to(device)
    u_true = u_true.to(device)
    v_true = v_true.to(device)

    # --- Adam warmup ---
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for it in range(adam_iters):
        optimizer.zero_grad()
        loss = pinn_loss(model, xyt, u_true, v_true)
        loss.backward()
        optimizer.step()
        if it % 100 == 0:
            print(f"Adam iter {it}, loss={loss.item():.3e}")

    # --- LBFGS fine-tuning ---
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_iters,
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        loss = pinn_loss(model, xyt, u_true, v_true)
        loss.backward()
        return loss

    print("Starting LBFGS optimization...")
    optimizer_lbfgs.step(closure)


# -----------------------------
# Helper function for 3D aspect
# -----------------------------
def axisEqual3D(ax):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    ax.set_xlim3d([centers[0] - r, centers[0] + r])
    ax.set_ylim3d([centers[1] - r, centers[1] + r])
    ax.set_zlim3d([centers[2] - r, centers[2] + r])


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Load Data
    data = scipy.io.loadmat("PINNs/main/Data/cylinder_nektar_wake.mat")
    U_star = data["U_star"]
    P_star = data["p_star"]
    t_star = data["t"]
    X_star = data["X_star"]

    N = X_star.shape[0]
    T = t_star.shape[0]

    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(t_star, (1, N)).T

    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star[:, 0]

    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]

    # Random training sample
    N_train = 5000
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = torch.tensor(x[idx, :], dtype=torch.float32)
    y_train = torch.tensor(y[idx, :], dtype=torch.float32)
    t_train = torch.tensor(t[idx, :], dtype=torch.float32)
    u_train = torch.tensor(u[idx, :], dtype=torch.float32)
    v_train = torch.tensor(v[idx, :], dtype=torch.float32)

    x_train.requires_grad_(True)
    y_train.requires_grad_(True)
    t_train.requires_grad_(True)

    # Stack inputs for CNN: [batch, channels, H, W]
    xyt_train = (
        torch.cat([x_train, y_train, t_train], dim=1).unsqueeze(-1).unsqueeze(-1)
    )
    xyt_train.requires_grad_(True)

    u_train = u_train.unsqueeze(-1).unsqueeze(-1)  # shape [N,1,1,1]
    v_train = v_train.unsqueeze(-1).unsqueeze(-1)

    # Build model
    model = PhysicsInformedCNN(in_channels=3, hidden_channels=32, out_channels=3)

    # Train
    train(model, xyt_train, u_train, v_train, adam_iters=500, lbfgs_iters=200)

    # Test example snapshot
    snap_idx = 100
    x_star_s = torch.tensor(X_star[:, 0:1], dtype=torch.float32)
    y_star_s = torch.tensor(X_star[:, 1:2], dtype=torch.float32)
    t_star_s = torch.tensor(TT[:, snap_idx], dtype=torch.float32).unsqueeze(1)


    xyt_test = (
        torch.cat([x_star_s, y_star_s, t_star_s], dim=1).unsqueeze(-1).unsqueeze(-1)
    )
    u_pred, v_pred, p_pred = model(xyt_test)
    print("u_pred.shape:", u_pred.shape)
    print("v_pred.shape:", v_pred.shape)

    p_star = P_star[:, snap_idx]

    # ----------------------------------------------------------------------
    # Noiseless Data
    # ----------------------------------------------------------------------
    # Training Data (already sampled)
    x_train_np = x_train.detach().cpu().numpy()
    y_train_np = y_train.detach().cpu().numpy()
    t_train_np = t_train.detach().cpu().numpy()

    u_train_np = u_train.detach().cpu().numpy()
    v_train_np = v_train.detach().cpu().numpy()
    p_train_np = p_pred.squeeze(-1).squeeze(-1).detach().cpu().numpy()

    # Train the model
    train(model, xyt_train, u_train, v_train, adam_iters=500, lbfgs_iters=200)

    # Test Data: snapshot
    snap_idx = 100
    x_star = torch.tensor(X_star[:, 0:1], dtype=torch.float32).to(device)
    y_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32).to(device)
    t_star = torch.tensor(TT[:, snap_idx], dtype=torch.float32).unsqueeze(1).to(device)

    xyt_test = torch.cat([x_star, y_star, t_star], dim=1).unsqueeze(-1).unsqueeze(-1)
    u_pred, v_pred, p_pred = model(xyt_test)

    # Convert predictions to numpy for comparison/plotting
    u_pred_np = u_pred.detach().cpu().numpy().squeeze()
    v_pred_np = v_pred.detach().cpu().numpy().squeeze()
    p_pred_np = p_pred.detach().cpu().numpy().squeeze()

    # True snapshot
    u_star = U_star[:, 0, snap_idx]
    v_star = U_star[:, 1, snap_idx]
    p_star = P_star[:, snap_idx]

    # Errors
    error_u = np.linalg.norm(u_star - u_pred_np, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred_np, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred_np, 2) / np.linalg.norm(p_star, 2)

    print(f"Error u: {error_u:.3e}")
    print(f"Error v: {error_v:.3e}")
    print(f"Error p: {error_p:.3e}")

    # ----------------------------------------------------------------------
    # Noisy Data
    # ----------------------------------------------------------------------
    noise = 0.01
    u_train_noisy = u_train_np + noise * np.std(u_train_np) * np.random.randn(
        *u_train_np.shape
    )
    v_train_noisy = v_train_np + noise * np.std(v_train_np) * np.random.randn(
        *v_train_np.shape
    )

    # Convert back to tensors
    u_train_noisy_t = (
        torch.tensor(u_train_noisy, dtype=torch.float32)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(device)
    )
    v_train_noisy_t = (
        torch.tensor(v_train_noisy, dtype=torch.float32)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(device)
    )

    # Train again with noisy data
    train(
        model,
        xyt_train,
        u_train_noisy_t,
        v_train_noisy_t,
        adam_iters=500,
        lbfgs_iters=200,
    )

    # Predict again
    u_pred_noisy, v_pred_noisy, p_pred_noisy = model(xyt_test)
    u_pred_noisy_np = u_pred_noisy.detach().cpu().numpy().squeeze()
    v_pred_noisy_np = v_pred_noisy.detach().cpu().numpy().squeeze()
    p_pred_noisy_np = p_pred_noisy.detach().cpu().numpy().squeeze()

    # Compute errors if desired
    error_u_noisy = np.linalg.norm(u_star - u_pred_noisy_np, 2) / np.linalg.norm(
        u_star, 2
    )
    error_v_noisy = np.linalg.norm(v_star - v_pred_noisy_np, 2) / np.linalg.norm(
        v_star, 2
    )
    error_p_noisy = np.linalg.norm(p_star - p_pred_noisy_np, 2) / np.linalg.norm(
        p_star, 2
    )

    print(f"Noisy Error u: {error_u_noisy:.3e}")
    print(f"Noisy Error v: {error_v_noisy:.3e}")
    print(f"Noisy Error p: {error_p_noisy:.3e}")

    # -----------------------------
    # Interpolation for plotting
    # -----------------------------
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x_grid = np.linspace(lb[0], ub[0], nn)
    y_grid = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x_grid, y_grid)

    UU_star = griddata(X_star, u_pred_np.flatten(), (X, Y), method="cubic")
    VV_star = griddata(X_star, v_pred_np.flatten(), (X, Y), method="cubic")
    PP_star = griddata(X_star, p_pred_np.flatten(), (X, Y), method="cubic")
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method="cubic")

    # -----------------------------
    # Plot 3D u and v fields
    # -----------------------------
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2)
    ax_u = fig.add_subplot(gs[0, 0], projection="3d")
    ax_v = fig.add_subplot(gs[0, 1], projection="3d")

    ax_u.plot_trisurf(X_star[:, 0], X_star[:, 1], u_pred_np, cmap="rainbow", alpha=0.8)
    ax_u.set_title("Predicted u(t,x,y)")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("y")

    ax_v.plot_trisurf(X_star[:, 0], X_star[:, 1], v_pred_np, cmap="rainbow", alpha=0.8)
    ax_v.set_title("Predicted v(t,x,y)")
    ax_v.set_xlabel("x")
    ax_v.set_ylabel("y")

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot pressure field (2D)
    # -----------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im0 = ax[0].imshow(
        PP_star,
        extent=[lb[0], ub[0], lb[1], ub[1]],
        origin="lower",
        cmap="rainbow",
        aspect="auto",
    )
    fig.colorbar(im0, ax=ax[0])
    ax[0].set_title("Predicted Pressure")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    im1 = ax[1].imshow(
        P_exact,
        extent=[lb[0], ub[0], lb[1], ub[1]],
        origin="lower",
        cmap="rainbow",
        aspect="auto",
    )
    fig.colorbar(im1, ax=ax[1])
    ax[1].set_title("Exact Pressure")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Print table of errors
    # -----------------------------
    print("Errors on clean data:")
    print(f"u: {error_u:.3e}, v: {error_v:.3e}, p: {error_p:.3e}")
    print("Errors on noisy data:")
    print(f"u: {error_u_noisy:.3e}, v: {error_v_noisy:.3e}, p: {error_p_noisy:.3e}")
