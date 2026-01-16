import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt  # Added for plotting

# -----------------------------
# Device Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(1234)
torch.manual_seed(1234)


# -----------------------------
# 1. Physics-Informed Model (Discovery Mode)
# -----------------------------
class PhysicsInformedDiscovery(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, out_channels=3):
        super().__init__()

        # --- Neural Network Layers ---
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        self.apply(self.init_weights)

        # --- Physics Parameters to Discover ---
        # Initialize to 0.0
        self.lambda1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.lambda2 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, xyt):
        h = torch.tanh(self.conv1(xyt))
        h = torch.tanh(self.conv2(h))
        h = torch.tanh(self.conv3(h))
        out = self.out_conv(h)
        return out[:, 0:1, :, :], out[:, 1:2, :, :], out[:, 2:3, :, :]


# -----------------------------
# 2. Physics Residual (Navier-Stokes)
# -----------------------------
def navier_stokes_residual(model, xyt):
    xyt.requires_grad_(True)

    u, v, p = model(xyt)

    l1 = model.lambda1
    l2 = model.lambda2

    # --- Gradients ---
    u_g = autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_y, u_t = u_g[:, 0:1], u_g[:, 1:2], u_g[:, 2:3]

    v_g = autograd.grad(v, xyt, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x, v_y, v_t = v_g[:, 0:1], v_g[:, 1:2], v_g[:, 2:3]

    p_g = autograd.grad(p, xyt, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_x, p_y = p_g[:, 0:1], p_g[:, 1:2]

    u_xx = autograd.grad(
        u_x, xyt, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0:1]
    u_yy = autograd.grad(
        u_y, xyt, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0][:, 1:2]

    v_xx = autograd.grad(
        v_x, xyt, grad_outputs=torch.ones_like(v_x), create_graph=True
    )[0][:, 0:1]
    v_yy = autograd.grad(
        v_y, xyt, grad_outputs=torch.ones_like(v_y), create_graph=True
    )[0][:, 1:2]

    # --- Physics Residuals ---
    f_u = u_t + l1 * (u * u_x + v * u_y) + p_x - l2 * (u_xx + u_yy)
    f_v = v_t + l1 * (u * v_x + v * v_y) + p_y - l2 * (v_xx + v_yy)

    return f_u, f_v, u, v


# -----------------------------
# 3. Training Loop with Plotting
# -----------------------------
def run_discovery_training(xyt_train, u_train, v_train, steps=5000):
    print(f"\n=== Starting Discovery Training (Steps={steps}) ===")

    model = PhysicsInformedDiscovery().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Lists to store history for plotting
    iter_history = []
    l1_history = []
    l2_history = []

    for it in range(steps):
        optimizer.zero_grad()

        f_u, f_v, u_pred, v_pred = navier_stokes_residual(model, xyt_train)

        mse_data = ((u_pred - u_train) ** 2).mean() + ((v_pred - v_train) ** 2).mean()
        mse_physics = (f_u**2).mean() + (f_v**2).mean()

        loss = mse_data + mse_physics
        loss.backward()
        optimizer.step()

        # Record and Print every 500 iterations
        if it % 500 == 0:
            l1_val = model.lambda1.item()
            l2_val = model.lambda2.item()

            iter_history.append(it)
            l1_history.append(l1_val)
            l2_history.append(l2_val)

            print(
                f"Iter {it:04d} | Loss: {loss.item():.5f} | "
                f"L1: {l1_val:.4f} | L2: {l2_val:.5f}"
            )

    iter_history.append(steps)
    l1_history.append(model.lambda1.item())
    l2_history.append(model.lambda2.item())

    # --- Plotting Results ---
    print("\n=== Plotting Discovery History ===")
    plt.figure(figsize=(12, 5))

    # Graph 1: Lambda 1 (Convection)
    plt.subplot(1, 2, 1)
    plt.plot(
        iter_history,
        l1_history,
        label="Predicted lambda_1",
        color="blue",
        linewidth=2,
    )
    plt.axhline(y=1.0, color="r", linestyle="--", label="True Value (1.0)")
    plt.xlabel("Iterations")
    plt.ylabel("Value of lambda_1")
    plt.title("Discovery of lambda_1 (Convection)")
    plt.legend()
    plt.grid(True)

    # Graph 2: Lambda 2 (Viscosity/Diffusion)
    plt.subplot(1, 2, 2)
    plt.plot(
        iter_history,
        l2_history,
        label="Predicted lambda_2",
        color="green",
        linewidth=2,
    )
    plt.axhline(
        y=0.01, color="r", linestyle="--", label="True Value (0.01)"
    )
    plt.xlabel("Iterations")
    plt.ylabel("Value of lambda_2")
    plt.title("Discovery of lambda_2 (Viscosity)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("discovery_evolution.png")  # Saves the plot
    plt.show()

    # Final Result
    print("\n=== Discovery Results ===")
    print(f"Final Lambda 1 (Expected ~1.0):  {model.lambda1.item():.5f}")
    print(f"Final Lambda 2 (Expected ~0.01): {model.lambda2.item():.5f}")

    return model


# -----------------------------
# 4. Main Execution
# -----------------------------

if __name__ == "__main__":
    data_path = "Data/cylinder_nektar_wake.mat"

    try:
        data = scipy.io.loadmat(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find file at {data_path}")
        exit()

    U_star = data["U_star"]
    t_star = data["t"]
    X_star = data["X_star"]

    N = X_star.shape[0]
    T = t_star.shape[0]

    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(t_star, (1, N)).T

    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]

    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]

    # Training Data Selection
    N_train = 5000
    idx = np.random.choice(N * T, N_train, replace=False)

    x_train = torch.tensor(x[idx, :], dtype=torch.float32)
    y_train = torch.tensor(y[idx, :], dtype=torch.float32)
    t_train = torch.tensor(t[idx, :], dtype=torch.float32)
    u_train = torch.tensor(u[idx, :], dtype=torch.float32)
    v_train = torch.tensor(v[idx, :], dtype=torch.float32)

    xyt_train = (
        torch.cat([x_train, y_train, t_train], dim=1).unsqueeze(-1).unsqueeze(-1)
    )
    u_train = u_train.unsqueeze(-1).unsqueeze(-1)
    v_train = v_train.unsqueeze(-1).unsqueeze(-1)

    xyt_train = xyt_train.to(device)
    u_train = u_train.to(device)
    v_train = v_train.to(device)

    # Run
    model = run_discovery_training(xyt_train, u_train, v_train, steps=5000)
