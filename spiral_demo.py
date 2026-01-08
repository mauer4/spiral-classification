# spiral_demo.py
# ------------------------------------------------------------
# Two-spiral growth experiment:
# 1) Start with a short (partial) spiral that is linearly separable.
# 2) Train a "neural network" with NO activation and NO bias:
#       p = sigmoid(X @ w)  -> decision boundary is X @ w = 0 (line through origin)
# 3) Gradually grow the spiral; show linear model fails once non-linearly separable.
# 4) Add a nonlinear activation (tanh/ReLU) in a small MLP; show it succeeds.
# 5) Continue growing; show nonlinear keeps working while linear cannot.
#
# Outputs:
#   - spiral_linear_vs_nonlinear.gif
#   - spiral_accuracy_summary.png
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# Set dark mode style
plt.style.use('dark_background')


# ----------------------------
# Data: two intertwined spirals
# ----------------------------
def make_two_spirals(n_per_class=250, theta_max=1.0, noise=0.03, seed=0):
    """
    Two classes in R^2.
    Class 0: spiral arm at angle theta
    Class 1: spiral arm phase-shifted by pi (opposite arm)

    theta sampled uniformly in [0, theta_max]
    radius r = theta (simple linear growth)
    """
    rng = np.random.default_rng(seed)

    theta0 = rng.uniform(0.0, theta_max, size=n_per_class)
    theta1 = rng.uniform(0.0, theta_max, size=n_per_class)

    r0 = theta0
    r1 = theta1

    x0 = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)
    x1 = np.stack([r1 * np.cos(theta1 + np.pi), r1 * np.sin(theta1 + np.pi)], axis=1)

    X = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])

    # Add isotropic Gaussian noise
    X = X + rng.normal(0.0, noise, size=X.shape)

    # Re-center around origin (helps match your "no bias needed" constraint)
    X = X - X.mean(axis=0, keepdims=True)
    
    # Filter out points too close to origin for clearer visualization
    min_radius = 0.15
    distances = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    mask = distances >= min_radius
    X = X[mask]
    y = y[mask]
    
    return X, y


# ----------------------------
# Core math helpers
# ----------------------------
def sigmoid(z):
    z = np.clip(z, -60, 60)  # avoid overflow
    return 1.0 / (1.0 + np.exp(-z))


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


# ---------------------------------------------------------
# Linear model: single layer, no activation, no bias
# Decision boundary: X @ w = 0  (line through origin)
# ---------------------------------------------------------
def train_linear_no_bias(X, y, lr=0.8, steps=2500, l2=2e-4, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.5, size=(2,))

    y01 = y.astype(float)

    for _ in range(steps):
        z = X @ w
        p = sigmoid(z)
        grad = (X.T @ (p - y01)) / X.shape[0] + l2 * w
        w -= lr * grad

    return w


def linear_predict(X, w):
    # threshold at 0 => line X @ w = 0
    return (X @ w >= 0).astype(int)


# ---------------------------------------------------------
# Nonlinear model: 2 -> hidden -> 1 with activation + bias
# ---------------------------------------------------------
def train_mlp(X, y, hidden=48, lr=0.4, steps=5000, l2=2e-4, seed=0, activation="tanh"):
    rng = np.random.default_rng(seed)

    # Xavier-ish init
    W1 = rng.normal(0.0, 1.0 / np.sqrt(2), size=(2, hidden))
    b1 = np.zeros((hidden,))
    W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden), size=(hidden,))
    b2 = 0.0

    y01 = y.astype(float)

    def act(z):
        if activation == "tanh":
            return np.tanh(z)
        if activation == "relu":
            return np.maximum(0.0, z)
        raise ValueError("activation must be 'tanh' or 'relu'")

    def dact(z):
        if activation == "tanh":
            a = np.tanh(z)
            return 1.0 - a * a
        if activation == "relu":
            return (z > 0.0).astype(float)

    for _ in range(steps):
        # Forward
        z1 = X @ W1 + b1          # (N,H)
        a1 = act(z1)              # (N,H)
        z2 = a1 @ W2 + b2         # (N,)
        p = sigmoid(z2)           # (N,)

        # Backward: logistic loss
        dz2 = (p - y01) / X.shape[0]     # (N,)
        dW2 = a1.T @ dz2 + l2 * W2       # (H,)
        db2 = dz2.sum()

        da1 = np.outer(dz2, W2)          # (N,H)
        dz1 = da1 * dact(z1)             # (N,H)
        dW1 = X.T @ dz1 + l2 * W1        # (2,H)
        db1 = dz1.sum(axis=0)            # (H,)

        # Update
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    return (W1, b1, W2, b2)


def mlp_predict(X, params, activation="tanh"):
    W1, b1, W2, b2 = params
    if activation == "tanh":
        a1 = np.tanh(X @ W1 + b1)
    elif activation == "relu":
        a1 = np.maximum(0.0, X @ W1 + b1)
    else:
        raise ValueError("activation must be 'tanh' or 'relu'")
    logits = a1 @ W2 + b2
    return (logits >= 0).astype(int)


# ----------------------------
# Plotting utilities
# ----------------------------
def plot_linear_boundary(ax, w, xlim, ylim, label=None, color='#00D9FF'):
    # w1 x + w2 y = 0 => y = -(w1/w2) x
    w1, w2 = float(w[0]), float(w[1])
    xs = np.linspace(xlim[0], xlim[1], 200)

    if abs(w2) < 1e-10:
        # vertical line x = 0
        h = ax.plot([0, 0], [ylim[0], ylim[1]], linewidth=2.5, label=label, color=color, alpha=0.9)
        return h[0]
    ys = -(w1 / w2) * xs
    h = ax.plot(xs, ys, linewidth=2.5, label=label, color=color, alpha=0.9)
    return h[0]


def plot_mlp_boundary(ax, params, xlim, ylim, activation="tanh", color='#FF10F0'):
    """
    Draw contour of logits=0. Return a proxy line handle for legend stability
    across matplotlib versions.
    """
    W1, b1, W2, b2 = params

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 260),
        np.linspace(ylim[0], ylim[1], 260),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    if activation == "tanh":
        a1 = np.tanh(grid @ W1 + b1)
    else:
        a1 = np.maximum(0.0, grid @ W1 + b1)

    logits = a1 @ W2 + b2
    zz = logits.reshape(xx.shape)

    ax.contour(xx, yy, zz, levels=[0.0], linewidths=2.5, colors=[color], alpha=0.9)

    # Proxy handle for legend with matching color
    return Line2D([0], [0], linewidth=2.5, color=color)


# ----------------------------
# Experiment driver
# ----------------------------
def run_spiral_experiment(
    n_per_class=250,
    noise=0.035,
    theta_start=0.7,
    theta_end=6.2,
    n_steps=18,
    seed=3,
    mlp_hidden=48,
    mlp_activation="tanh",
    out_gif=None,
    out_png=None,
):
    # Auto-generate output filenames if not provided
    if out_gif is None:
        out_gif = f"spiral_linear_vs_nonlinear_h{mlp_hidden}.gif"
    if out_png is None:
        out_png = f"spiral_accuracy_summary_h{mlp_hidden}.png"
    
    # Spiral growth schedule
    thetas = np.linspace(theta_start, theta_end, n_steps)

    # For consistent axes across frames, compute global bounds at max theta
    X_all, _ = make_two_spirals(n_per_class, theta_max=theta_end, noise=noise, seed=seed)
    pad = 0.35
    xlim = (X_all[:, 0].min() - pad, X_all[:, 0].max() + pad)
    ylim = (X_all[:, 1].min() - pad, X_all[:, 1].max() + pad)

    # Metrics for summary
    lin_accs = np.zeros(n_steps, dtype=float)
    mlp_accs = np.zeros(n_steps, dtype=float)
    
    # Pre-compute to find when linear model first fails
    first_fail_idx = None
    for i in range(n_steps):
        theta_max = float(thetas[i])
        X_temp, y_temp = make_two_spirals(n_per_class, theta_max, noise, seed + i)
        w_temp = train_linear_no_bias(X_temp, y_temp, lr=0.8, steps=2500, l2=2e-4, seed=seed + 10 * i)
        y_lin_temp = linear_predict(X_temp, w_temp)
        acc_temp = accuracy(y_temp, y_lin_temp)
        if first_fail_idx is None and acc_temp < 0.85:
            first_fail_idx = i
            break

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    def update(i):
        ax.clear()

        theta_max = float(thetas[i])
        X, y = make_two_spirals(
            n_per_class=n_per_class,
            theta_max=theta_max,
            noise=noise,
            seed=seed + i,
        )

        # Train linear model (no bias)
        w = train_linear_no_bias(
            X, y,
            lr=0.8,
            steps=2500,
            l2=2e-4,
            seed=seed + 10 * i,
        )
        y_lin = linear_predict(X, w)
        acc_lin = accuracy(y, y_lin)

        # Train nonlinear MLP
        params = train_mlp(
            X, y,
            hidden=mlp_hidden,
            lr=0.4,
            steps=5000,
            l2=2e-4,
            seed=seed + 999 + 10 * i,
            activation=mlp_activation,
        )
        y_mlp = mlp_predict(X, params, activation=mlp_activation)
        acc_mlp = accuracy(y, y_mlp)

        lin_accs[i] = acc_lin
        mlp_accs[i] = acc_mlp

        # Scatter points with better colors for dark mode
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=18, alpha=0.8, 
                  color='#FF6B35', edgecolors='white', linewidth=0.3)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, alpha=0.8,
                  color='#4ECDC4', edgecolors='white', linewidth=0.3)

        # Boundaries
        plot_linear_boundary(
            ax, w, xlim, ylim,
            label=f"Linear perceptron - 2 weights"
        )
        mlp_proxy = plot_mlp_boundary(ax, params, xlim, ylim, activation=mlp_activation)

        # Legend: use proxy for contour boundary
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mlp_proxy)
        labels.append(f"MLP with non-linear activation function")
        ax.legend(handles, labels, loc="upper right", framealpha=0.9, fontsize=10)

        # Cosmetics
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Two-spiral growth: Separable by line?", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.grid(True, alpha=0.15, linestyle='--')
        
        return (w, params, acc_lin, acc_mlp, X, y)
    
    def update_warning(flash_num, base_data):
        """Create warning flash frames with arrows pointing to misclassified points"""
        w, params, acc_lin, acc_mlp, X, y = base_data
        ax.clear()
        
        theta_max = float(thetas[first_fail_idx])
        
        # Scatter points with dark mode colors
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=18, alpha=0.8,
                  color='#FF6B35', edgecolors='white', linewidth=0.3)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, alpha=0.8,
                  color='#4ECDC4', edgecolors='white', linewidth=0.3)
        
        # Boundaries
        plot_linear_boundary(ax, w, xlim, ylim, label=f"Linear (no act, no bias) acc={acc_lin:.2f}")
        mlp_proxy = plot_mlp_boundary(ax, params, xlim, ylim, activation=mlp_activation)
        
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mlp_proxy)
        labels.append(f"MLP (+{mlp_activation}) acc={acc_mlp:.2f}")
        ax.legend(handles, labels, loc="upper right", framealpha=0.9, fontsize=10)
        
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Two-spiral growth: theta_max={theta_max:.2f}", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.grid(True, alpha=0.15, linestyle='--')
        
        # Find misclassified points (where linear boundary crosses the wrong class)
        y_pred_lin = linear_predict(X, w)
        misclassified = (y != y_pred_lin)
        
        # Flash warning on alternate frames
        if flash_num % 2 == 0:
            ax.text(0.5, 0.5, "⚠ NEED\nNON-LINEAR ⚠",
                   transform=ax.transAxes,
                   fontsize=38, fontweight='bold', family='monospace',
                   ha='center', va='center',
                   color='#FF3333', alpha=0.95,
                   bbox=dict(boxstyle='round,pad=1.0', facecolor='#FFD700', alpha=0.9, 
                            edgecolor='#FF3333', linewidth=4),
                   rotation=-12,
                   zorder=100)

    # Create frame sequence with warning flashes
    warning_base_data = None
    def combined_update(frame_idx):
        nonlocal warning_base_data
        if first_fail_idx is not None and frame_idx == first_fail_idx:
            # Store base data for warning frames
            warning_base_data = update(frame_idx)
        elif first_fail_idx is not None and frame_idx > first_fail_idx and frame_idx <= first_fail_idx + 4:
            # Show warning flashes
            update_warning(frame_idx - first_fail_idx - 1, warning_base_data)
        else:
            # Normal frames
            update(frame_idx if frame_idx <= first_fail_idx else frame_idx - 4)
    
    # Add 4 extra frames for warning if linear model fails
    total_frames = n_steps + (4 if first_fail_idx is not None else 0)
    
    anim = FuncAnimation(fig, combined_update, frames=total_frames, interval=650, repeat=False)

    # Save animation; if something unexpected fails, still save a fallback frame
    try:
        anim.save(out_gif, writer=PillowWriter(fps=1.3))
        print(f"Saved animation: {out_gif}")
    except Exception as e:
        print("Animation save failed:", repr(e))
        fig.savefig("spiral_last_frame.png", dpi=160)
        print("Saved fallback frame: spiral_last_frame.png")

    plt.close(fig)

    # Summary plot with dark mode styling
    fig2, ax2 = plt.subplots(figsize=(10, 5.5), facecolor='#1a1a1a')
    ax2.set_facecolor('#1a1a1a')
    ax2.plot(thetas, lin_accs, marker="o", label="Linear (no activation, no bias)", 
            linewidth=2.5, markersize=8, color='#00D9FF')
    ax2.plot(thetas, mlp_accs, marker="o", label=f"MLP (+{mlp_activation})",
            linewidth=2.5, markersize=8, color='#FF10F0')
    ax2.set_xlabel("Spiral length (theta_max)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Training accuracy", fontsize=12, fontweight='bold')
    ax2.set_title("Accuracy vs spiral growth: linear fails, nonlinear succeeds", 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linewidth=0.5, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, framealpha=0.9, loc='lower left')
    ax2.tick_params(labelsize=10)
    fig2.tight_layout()
    fig2.savefig(out_png, dpi=160, facecolor='#1a1a1a')
    plt.close(fig2)

    print(f"Saved summary plot: {out_png}")


if __name__ == "__main__":
    run_spiral_experiment(
        n_per_class=250,
        noise=0.035,
        theta_start=0.7,   # usually separable early
        theta_end=6.2,     # clearly non-separable later
        n_steps=18,
        seed=3,
        mlp_hidden=4,
        mlp_activation="tanh",  # change to "relu" if you want
        # out_gif and out_png will be auto-generated as h4 files
    )
