import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sophia import SophiaG

# Create log directory
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set double precision
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random seed
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


# Define PINN model for Helmholtz equation
class PINN(nn.Module):
    def __init__(self, k=2.0):  # Set wave number to 2.0
        super(PINN, self).__init__()
        # Network structure: 2 input features (x,y), 1 output value (u)
        self.net = MLP([2, 64, 64, 64, 64, 1])
        self.k = k  # Wave number
        self.pi = torch.tensor(np.pi)

    def forward(self, x, y):
        # Combine inputs as tensor
        xy = torch.cat([x, y], dim=1)
        return self.net(xy)

    def f(self, x, y):
        """Calculate Helmholtz equation residual: ∇²u + k²u = 0"""
        x.requires_grad_(True)
        y.requires_grad_(True)

        u = self.forward(x, y)

        # Calculate second-order derivative with respect to x
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate second-order derivative with respect to y
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]

        # Helmholtz equation: ∇²u + k²u = 0
        residual = u_xx + u_yy + (self.k**2) * u

        return residual


# Analytical solution for Helmholtz equation with specific boundary conditions
def exact_solution(x, y):
    """
    Analytical solution: u(x,y) = sin(πx)sin(πy)
    For this solution with wave number k=2, we need: k² = 4 = π² + π²
    Therefore π² = 2, so π = √2
    We use sin(√2·x)sin(√2·y) as our analytical solution
    """
    scaling_factor = np.sqrt(2)
    return torch.sin(scaling_factor * x) * torch.sin(scaling_factor * y)


# Generate training data for Helmholtz equation
def generate_training_data(n_points, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
    # Interior points
    x_domain = torch.rand(n_points, 1, device=device) * (x_max - x_min) + x_min
    y_domain = torch.rand(n_points, 1, device=device) * (y_max - y_min) + y_min

    # Boundary points (x=0, x=1, y=0, y=1)
    # x=0 boundary
    y_boundary_x0 = torch.rand(n_points // 4, 1, device=device) * (y_max - y_min) + y_min
    x_boundary_x0 = torch.zeros(n_points // 4, 1, device=device)
    
    # x=1 boundary
    y_boundary_x1 = torch.rand(n_points // 4, 1, device=device) * (y_max - y_min) + y_min
    x_boundary_x1 = torch.ones(n_points // 4, 1, device=device)
    
    # y=0 boundary
    x_boundary_y0 = torch.rand(n_points // 4, 1, device=device) * (x_max - x_min) + x_min
    y_boundary_y0 = torch.zeros(n_points // 4, 1, device=device)
    
    # y=1 boundary
    x_boundary_y1 = torch.rand(n_points // 4, 1, device=device) * (x_max - x_min) + x_min
    y_boundary_y1 = torch.ones(n_points // 4, 1, device=device)
    
    # Combine boundary points
    x_boundary = torch.cat([x_boundary_x0, x_boundary_x1, x_boundary_y0, x_boundary_y1])
    y_boundary = torch.cat([y_boundary_x0, y_boundary_x1, y_boundary_y0, y_boundary_y1])
    
    return x_domain, y_domain, x_boundary, y_boundary


# Training function
def train(model, inputs, n_epochs):
    # Get training data
    x_domain, y_domain, x_boundary, y_boundary = inputs

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # Uncomment to use alternative optimizers
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # optimizer = optim.NAdam(model.parameters(), lr=1e-3)
    # optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    # optimizer = SophiaG(model.parameters(), lr=1e-4)

    # Training loop
    losses = []
    for epoch in range(n_epochs):
        # Calculate PDE residual loss
        f_pred = model.f(x_domain, y_domain)
        loss_f = torch.mean(torch.square(f_pred))

        # Calculate boundary condition loss (u=0 at all boundaries)
        u_boundary = model(x_boundary, y_boundary)
        loss_bc = torch.mean(torch.square(u_boundary))

        # Total loss
        loss = loss_f + loss_bc

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


# Evaluate and visualize results
def evaluate_model(model, n_points=100):
    # Create grid points
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X.flatten()[:, None], device=device)
    Y_tensor = torch.tensor(Y.flatten()[:, None], device=device)

    # Predict
    model.eval()
    with torch.no_grad():
        u_pred = model(X_tensor, Y_tensor).cpu().numpy()

    # Calculate analytical solution
    u_exact = exact_solution(X_tensor, Y_tensor).cpu().numpy()

    # Calculate error
    error = np.abs(u_pred - u_exact)

    # Reshape to grid shape
    U_pred = u_pred.reshape(n_points, n_points)
    U_exact = u_exact.reshape(n_points, n_points)
    Error = error.reshape(n_points, n_points)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Predicted solution
    im1 = axes[0].imshow(U_pred, cmap='viridis', origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    axes[0].set_title('PINN Predicted Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])

    # Exact solution
    im2 = axes[1].imshow(U_exact, cmap='viridis', origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    axes[1].set_title('Exact Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])

    # Error
    im3 = axes[2].imshow(Error, cmap='jet', origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig('./logs/helmholtz_adam_results.png', dpi=300)
    plt.show()

    # Calculate relative L2 error
    l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    print(f'Relative L2 Error: {l2_error:.6e}')

    return U_pred, U_exact, Error, l2_error


# Main function
def main():
    # Create model with wave number k=2
    k = 2.0
    model = PINN(k=k).to(device)

    # Prepare input data
    inputs = generate_training_data(n_points=1000)

    # Train model
    print("Starting training...")
    losses = train(model, inputs, n_epochs=20400)

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value (Log Scale)')
    plt.grid(True)
    plt.savefig('./logs/helmholtz_adam_loss.png', dpi=300)
    plt.show()

    # Evaluate model
    print("Evaluating model...")
    U_pred, U_exact, Error, l2_error = evaluate_model(model)


if __name__ == "__main__":
    main()