import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Create logs directory
if not os.path.exists('./logs'):
    os.makedirs('./logs')

# Set double precision
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# Define neural network model
class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Tanh()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.activation = activation

        # Initialize weights
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


# Define PINN model for KdV equation
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Define network structure: input 2 features (x,t), output 1 value (u)
        self.net = MLP([2, 128, 1])

    def forward(self, x, t):
        # Concatenate inputs into tensor
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

    def f(self, x, t):
        """Calculate KdV equation residual: ∂u/∂t + u∂u/∂x + ∂³u/∂x³ = 0"""
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = self.forward(x, t)

        # Calculate first-order derivative with respect to time
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate first-order derivative with respect to x
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate second-order derivative with respect to x
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate third-order derivative with respect to x
        u_xxx = torch.autograd.grad(
            u_xx, x,
            grad_outputs=torch.ones_like(u_xx),
            retain_graph=True,
            create_graph=True
        )[0]

        # KdV equation: ∂u/∂t + 6u∂u/∂x + ∂³u/∂x³ = 0
        residual = u_t + 6 * u * u_x + u_xxx

        return residual


# Load data from CSV file
def load_data():
    sim = pd.read_csv('./data/kdv.csv')
    sim['x'], sim['t'] = sim['x'], sim['t']
    x_train = np.vstack([sim.x.values, sim.t.values]).T
    y_train = sim[['u']].values
    
    # Filter data for specified domain
    domain = (x_train[:,0] >= 0.0) & (x_train[:,0] <= 1.5)
    x_train, y_train = x_train[domain], y_train[domain]
    
    return x_train, y_train


# Generate training data
def generate_training_data(x_min=0.0, x_max=1.5, t_min=0.0, t_max=2.0, nx=200, nt=100):
    # Load CSV data
    x_train, y_train = load_data()
    
    # Extract coordinates
    x = x_train[:,0:1].reshape(-1,1)
    t = x_train[:,1:2].reshape(-1,1)
    
    # Separate non-initial and initial condition data
    _train1 = np.argwhere(t != 0.0)[:, 0]
    x_train1 = x[_train1].reshape(-1,1)
    t_train1 = t[_train1].reshape(-1,1)
    y_train1 = y_train[_train1].reshape(-1,1)

    _train2 = np.argwhere(t == 0.0)[:, 0]
    x_train2 = x[_train2].reshape(-1,1)
    t_train2 = t[_train2].reshape(-1,1)
    y_train2 = y_train[_train2].reshape(-1,1)
    
    # Convert to PyTorch tensors
    x_domain = torch.tensor(x_train1, dtype=torch.float64, device=device)
    t_domain = torch.tensor(t_train1, dtype=torch.float64, device=device)
    
    x_initial = torch.tensor(x_train2, dtype=torch.float64, device=device)
    t_initial = torch.tensor(t_train2, dtype=torch.float64, device=device)
    u_initial = torch.tensor(y_train2, dtype=torch.float64, device=device)
    
    # Boundary points (x=x_min, x=x_max)
    N_b = 100
    t_b = torch.rand(N_b, 1, device=device) * (t_max - t_min) + t_min
    x_lb = torch.full_like(t_b, x_min)
    x_ub = torch.full_like(t_b, x_max)
    
    # Interior collocation points
    N_f = 5000
    x_f = torch.rand(N_f, 1, device=device) * (x_max - x_min) + x_min
    t_f = torch.rand(N_f, 1, device=device) * (t_max - t_min) + t_min
    
    # Create grid for visualization
    x_vis = np.linspace(x_min, x_max, nx)[:, None]
    t_vis = np.linspace(t_min, t_max, nt)[:, None]
    X, T = np.meshgrid(x_vis.flatten(), t_vis.flatten())
    
    # Prepare exact solution
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    
    # Find closest data points
    Exact = np.zeros(X_star.shape[0])
    for i, (x_i, t_i) in enumerate(X_star):
        # Find closest training data point
        idx = np.argmin(np.sum(np.square(x_train - np.array([x_i, t_i])), axis=1))
        Exact[i] = y_train[idx]
    
    Exact = Exact.reshape(X.shape)
    
    return x_lb, t_b, x_ub, t_b, x_initial, t_initial, u_initial, x_f, t_f, x_vis, t_vis, Exact, X, T


# Training function
def train(model, n_epochs, x_lb, t_lb, x_ub, t_ub, x_0, t_0, u_0, x_f, t_f):
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    losses = []

    for epoch in range(n_epochs):
        # Calculate PDE residual loss
        f_pred = model.f(x_f, t_f)
        loss_f = torch.mean(torch.square(f_pred))

        # Calculate boundary condition loss (periodic boundary)
        u_lb = model(x_lb, t_lb)
        u_ub = model(x_ub, t_ub)
        loss_bc = torch.mean(torch.square(u_lb - u_ub))

        # Calculate initial condition loss
        u_0_pred = model(x_0, t_0)
        loss_ic = torch.mean(torch.square(u_0_pred - u_0))

        # Total loss
        loss = loss_f + 10.0 * loss_ic + loss_bc

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        losses.append(loss.item())

        # Print training progress
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6e}')

    return losses


# Evaluate and visualize results
def evaluate_model(model, x, t, Exact, X, T):
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X.flatten()[:, None], device=device)
    T_tensor = torch.tensor(T.flatten()[:, None], device=device)

    # Predict
    model.eval()
    with torch.no_grad():
        u_pred = model(X_tensor, T_tensor).cpu().numpy()

    # Reshape to grid shape
    U_pred = u_pred.reshape(T.shape)

    # Calculate error
    error_u = np.linalg.norm(Exact.flatten() - U_pred.flatten(), 2) / np.linalg.norm(Exact.flatten(), 2)
    print(f'Relative L2 error: {error_u:.6e}')

    # Visualization - 2D plots only
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Select time slices for visualization
    t_slices = [0, 25, 50, 75]  # Indices for different time points
    
    # Plot exact solution at different time slices
    for i, t_idx in enumerate(t_slices):
        if i < 2:
            row, col = 0, i
        else:
            row, col = 1, i-2
            
        axs[row, col].plot(x.flatten(), Exact[t_idx, :], 'b-', linewidth=2, label='Exact')
        axs[row, col].plot(x.flatten(), U_pred[t_idx, :], 'r--', linewidth=2, label='PINN')
        axs[row, col].set_xlabel('x')
        axs[row, col].set_ylabel('u(x,t)')
        axs[row, col].set_title(f't = {t[t_idx][0]:.2f}')
        axs[row, col].grid(True)
        axs[row, col].legend()
    
    plt.tight_layout()
    plt.savefig('./logs/kdv_solution_slices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2D heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Exact solution heatmap
    im0 = axs[0].imshow(Exact, interpolation='nearest', cmap='rainbow',
                  extent=[x.min(), x.max(), t.min(), t.max()],
                  origin='lower', aspect='auto')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('t')
    axs[0].set_title('Exact Solution')
    fig.colorbar(im0, ax=axs[0])
    
    # PINN prediction heatmap
    im1 = axs[1].imshow(U_pred, interpolation='nearest', cmap='rainbow',
                  extent=[x.min(), x.max(), t.min(), t.max()],
                  origin='lower', aspect='auto')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('t')
    axs[1].set_title('PINN Prediction')
    fig.colorbar(im1, ax=axs[1])
    
    plt.tight_layout()
    plt.savefig('./logs/kdv_solution_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    return U_pred, Exact, error_u


def main():
    # Create model
    model = PINN().to(device)

    # Generate training data
    x_lb, t_lb, x_ub, t_ub, x_0, t_0, u_0, x_f, t_f, x_tensor, t_tensor, Exact, X, T = generate_training_data()

    # Train model
    print("Starting training...")
    losses = train(model, n_epochs=10000,
                   x_lb=x_lb, t_lb=t_lb,
                   x_ub=x_ub, t_ub=t_ub,
                   x_0=x_0, t_0=t_0, u_0=u_0,
                   x_f=x_f, t_f=t_f)

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.grid(True)
    plt.savefig('./logs/kdv_loss.png', dpi=300)
    plt.show()

    # Evaluate model
    print("Evaluating model...")
    U_pred, Exact, error_u = evaluate_model(model, x_tensor, t_tensor, Exact, X, T)


if __name__ == "__main__":
    main()