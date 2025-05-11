import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc, norm

from utils import get_model_params, set_model_params

# Create logs directory if it doesn't exist
if not os.path.exists('./logs'):
    os.makedirs('./logs')

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


# Define PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Define network structure: input 2 features (x,y), output 1 value (u)
        self.net = MLP([2, 32, 1])
        self.pi = torch.tensor(np.pi)

    def forward(self, x, y):
        # Combine inputs into tensor
        xy = torch.cat([x, y], dim=1)
        return self.net(xy)

    def f(self, x, y):
        """Calculate Poisson equation residual: ∇²u + f = 0, where f = 2π²sin(πx)sin(πy)"""
        x.requires_grad_(True)
        y.requires_grad_(True)

        u = self.forward(x, y)

        # Calculate first derivatives
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate second derivatives
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]

        # Poisson equation source term
        f = 2 * (self.pi ** 2) * torch.sin(self.pi * x) * torch.sin(self.pi * y)

        # Residual: ∇²u + f = 0 => ∇²u = -f => u_xx + u_yy = -f
        residual = u_xx + u_yy + f

        return residual


# Analytical solution
def exact_solution(x, y):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)


# Loss function calculation
def loss_fun(model, params, inputs, target):
    # Save original parameters
    original_params = get_model_params(model).clone()

    # Set new parameters
    params_tensor = torch.tensor(params, dtype=torch.float64, device=device)
    set_model_params(model, params_tensor)

    # Calculate loss
    x_domain, y_domain, x_boundary, y_boundary = inputs

    # Calculate PDE residual loss
    f_pred = model.f(x_domain, y_domain)
    loss_f = torch.mean(torch.square(f_pred))

    # Calculate boundary condition loss
    u_boundary = model(x_boundary, y_boundary)
    loss_bc = torch.mean(torch.square(u_boundary))

    # Total loss
    loss = loss_f + loss_bc

    # Restore original parameters
    set_model_params(model, original_params)

    return loss.item()


# APP optimization algorithm
def app(model, inputs, target, K, lambda_, rho, n):
    params = get_model_params(model)
    d = len(params)
    xk = params.detach().cpu().numpy().astype(np.float64)  # Added .cpu() before .numpy()
    loss_history = []
    alpha = lambda_

    halton = qmc.Halton(d=d, scramble=True, seed=42)
    fc = np.inf
    for i in range(K):
        # Generate n random vectors from Halton sequence
        x = halton.random(n)
        t = np.vstack([xk, norm.ppf(x, loc=xk, scale=1 / alpha)])

        # Compute function value sequence
        f = [loss_fun(model, t[k], inputs, target) for k in range(n + 1)]
        fk = f[0]
        f_min = min(f)
        fc = min(fc, f_min)
        f = np.array(f) - f_min

        # Use averaged asymptotic formula
        f_mean = np.mean(f)
        if f_mean > 0:
            f /= f_mean

        # Compute weights and new xk
        weights = np.exp(-f)
        xk = np.average(t, axis=0, weights=weights)

        # Update parameters and record
        set_model_params(model, torch.tensor(xk, dtype=torch.float64, device=device))
        alpha /= rho
        loss_history.append(fk)
        print(f'APP - Epoch {i + 1}, Loss: {fk:.6e}')

    return loss_history


# Training function
def train(model, n_points=100, K=100, lambda_=1.0, rho=1.1, n=10):
    # Create training data
    # Interior points
    x_domain = torch.rand(n_points, 1, device=device)
    y_domain = torch.rand(n_points, 1, device=device)

    # Boundary points (x=0, x=1, y=0, y=1)
    x_boundary_0 = torch.zeros(n_points // 4, 1, device=device)
    y_boundary_0 = torch.rand(n_points // 4, 1, device=device)

    x_boundary_1 = torch.ones(n_points // 4, 1, device=device)
    y_boundary_1 = torch.rand(n_points // 4, 1, device=device)

    x_boundary_2 = torch.rand(n_points // 4, 1, device=device)
    y_boundary_2 = torch.zeros(n_points // 4, 1, device=device)

    x_boundary_3 = torch.rand(n_points // 4, 1, device=device)
    y_boundary_3 = torch.ones(n_points // 4, 1, device=device)

    # Merge all boundary points
    x_boundary = torch.cat([x_boundary_0, x_boundary_1, x_boundary_2, x_boundary_3])
    y_boundary = torch.cat([y_boundary_0, y_boundary_1, y_boundary_2, y_boundary_3])

    # Use APP optimization algorithm
    inputs = (x_domain, y_domain, x_boundary, y_boundary)
    target = None  # No target values needed for this problem

    return app(model, inputs, target, K, lambda_, rho, n)


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
    im1 = axes[0].imshow(U_pred, cmap='viridis')
    axes[0].set_title('PINN Predicted Solution')
    plt.colorbar(im1, ax=axes[0])

    # Exact solution
    im2 = axes[1].imshow(U_exact, cmap='viridis')
    axes[1].set_title('Exact Solution')
    plt.colorbar(im2, ax=axes[1])

    # Error
    im3 = axes[2].imshow(Error, cmap='jet')
    axes[2].set_title('Absolute Error')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig('./logs/poisson_app_results.png', dpi=300)
    plt.show()

    # Calculate L2 relative error
    l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    print(f'L2 relative error: {l2_error:.6e}')

    return U_pred, U_exact, Error, l2_error


# Main function
def main():
    # Create model
    model = PINN().to(device)

    # Use APP optimization algorithm
    print("Starting training...")
    losses = train(
        model,
        n_points=400,
        K=200,
        lambda_=1 / np.sqrt(len(get_model_params(model))),
        rho=0.98,
        n=len(get_model_params(model))
    )

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value (Log Scale)')
    plt.grid(True)
    plt.savefig('./logs/poisson_app_loss.png', dpi=300)
    plt.show()

    # Evaluate model
    print("Evaluating model...")
    U_pred, U_exact, Error, l2_error = evaluate_model(model)

    # Save model
    # torch.save(model.state_dict(), './logs/poisson_app_model.pt')
    # print("Model saved as './logs/poisson_app_model.pt'")


if __name__ == "__main__":
    main()
