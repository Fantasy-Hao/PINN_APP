import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sophia import SophiaG


# Create logs directory
if not os.path.exists('./logs'):
    os.makedirs('./logs')

# Set double precision
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random seed for reproducibility
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


# Define PINN model for nonlinear equation
class PINN(nn.Module):
    def __init__(self, vis=0.001):
        super(PINN, self).__init__()
        # Define network structure: input 2 features (x,t), output 1 value (u)
        self.net = MLP([2, 64, 1])
        self.vis = vis  # Viscosity coefficient

    def forward(self, x, t):
        # Normalize inputs: x from [-2,2] to [0,1], t from [0,2] to [0,1]
        x_norm = (x + 2.0) / 4.0
        t_norm = t / 2.0
        
        # Combine normalized inputs into tensor
        xt = torch.cat([x_norm, t_norm], dim=1)
        return self.net(xt)

    def f(self, x, t):
        """Calculate nonlinear equation residual: ∂u/∂t + u∂u/∂x - vis∂²u/∂x² = 0"""
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = self.forward(x, t)

        # Calculate first derivative with respect to time
        # Scale factor 1/2.0 for t normalization [0,2] -> [0,1]
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0] / 2.0

        # Calculate first derivative with respect to x
        # Scale factor 1/4.0 for x normalization [-2,2] -> [0,1]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0] / 4.0

        # Calculate second derivative with respect to x
        # Additional scale factor 1/4.0 for second derivative
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0] / 4.0

        # Nonlinear equation: ∂u/∂t + u∂u/∂x - vis∂²u/∂x² = 0
        residual = u_t + u * u_x - self.vis * u_xx

        return residual


# Initial condition function
def f_ic(x, k=2):
    return torch.exp(-(k * x) ** 2)


# Generate training data for nonlinear equation
def generate_training_data(n_points, x_min=-2.0, x_max=2.0, t_min=0.0, t_max=2.0, vis=0.001):
    # Load simulation data for reference
    sim = pd.read_csv('./data/nonlinear.csv')

    # Interior points (domain points)
    x_domain = torch.rand(n_points, 1, device=device) * (x_max - x_min) + x_min
    t_domain = torch.rand(n_points, 1, device=device) * (t_max - t_min) + t_min

    # Initial condition points (t=0)
    x_initial = torch.rand(n_points // 4, 1, device=device) * (x_max - x_min) + x_min
    t_initial = torch.zeros(n_points // 4, 1, device=device)

    # Calculate initial condition values
    u_initial = f_ic(x_initial)

    return x_domain, t_domain, x_initial, t_initial, u_initial, sim


def train(model, inputs, n_epochs):
    # Unpack input data
    x_domain, t_domain, x_initial, t_initial, u_initial = inputs

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # optimizer = optim.NAdam(model.parameters(), lr=1e-3)
    # optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    # optimizer = SophiaG(model.parameters(), lr=1e-4)

    # Training loop
    losses = []
    for epoch in range(n_epochs):
        # Calculate PDE residual loss
        f_pred = model.f(x_domain, t_domain)
        loss_pde = torch.mean(torch.square(f_pred))

        # Calculate initial condition loss
        u_ic_pred = model(x_initial, t_initial)
        loss_ic = torch.mean(torch.square(u_ic_pred - u_initial))

        # Total loss
        loss = loss_pde + loss_ic

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        losses.append(loss.item())

        # Print training progress
        if epoch % 1000 == 0:
            print(f'Adam - Epoch {epoch}, Loss: {loss.item():.6e}')

    return losses


# Evaluate and visualize results using the entire dataset
def evaluate_model(model, sim):
    # Extract data from simulation
    x_sim = sim.x.values
    t_sim = sim.t.values
    u_sim = sim.u.values

    # Convert to PyTorch tensors for prediction
    x_tensor = torch.tensor(x_sim.reshape(-1, 1), device=device)
    t_tensor = torch.tensor(t_sim.reshape(-1, 1), device=device)

    # Predict using the model
    model.eval()
    with torch.no_grad():
        u_pred = model(x_tensor, t_tensor).cpu().numpy().flatten()

    # Calculate error
    error_u = np.linalg.norm(u_sim - u_pred, 2) / np.linalg.norm(u_sim, 2)
    print(f'Relative L2 error: {error_u:.6e}')

    # Create unique x and t values for plotting
    x_unique = np.unique(x_sim)
    t_unique = np.unique(t_sim)

    # Create 2D arrays for visualization
    X, T = np.meshgrid(x_unique, t_unique)
    U_pred = np.zeros(X.shape)
    U_sim = np.zeros(X.shape)

    # Fill the arrays with predicted and true values
    for i, t_val in enumerate(t_unique):
        for j, x_val in enumerate(x_unique):
            # Find indices where x=x_val and t=t_val
            idx = np.where((x_sim == x_val) & (t_sim == t_val))[0]
            if len(idx) > 0:
                # Get the first matching index
                idx = idx[0]
                U_pred[i, j] = u_pred[idx]
                U_sim[i, j] = u_sim[idx]

    # Visualization
    fig = plt.figure(figsize=(12, 5))

    # 2D heatmap - predicted solution
    ax = fig.add_subplot(121)
    h = ax.imshow(U_pred, interpolation='nearest', cmap='rainbow',
                  extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
                  origin='lower', aspect='auto')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_title('PINN Prediction')
    fig.colorbar(h, ax=ax)

    # 2D heatmap - exact solution
    ax = fig.add_subplot(122)
    h = ax.imshow(U_sim, interpolation='nearest', cmap='rainbow',
                  extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
                  origin='lower', aspect='auto')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_title('Reference Solution')
    fig.colorbar(h, ax=ax)

    plt.tight_layout()
    plt.savefig('./logs/nonlinear_solution.png', dpi=300, bbox_inches='tight')
    plt.show()

    return u_pred, error_u


# Main function
def main():
    # Create model
    vis = 0.001
    model = PINN(vis=vis).to(device)

    # Generate training data
    n_points = 1000
    x_domain, t_domain, x_initial, t_initial, u_initial, sim = generate_training_data(n_points=n_points, vis=vis)

    # Package inputs for training
    inputs = (x_domain, t_domain, x_initial, t_initial, u_initial)

    # Train using Adam optimizer
    print("Training with Adam optimizer...")
    losses = train(model, inputs=inputs, n_epochs=10200)

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.semilogy(losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.grid(True)
    plt.savefig('./logs/nonlinear_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Evaluate model
    print("Evaluating model...")
    u_pred, error_u = evaluate_model(model, sim)


if __name__ == "__main__":
    main()