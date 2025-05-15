"""
APP + L-BFGS Optimization.
Heat equation example. Solution given by

u(x,t) = sin(pi*x) * exp(-pi^2*t).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc, norm

from utils import get_model_params, set_model_params

# Create logs directory if it doesn't exist
if not os.path.exists('../logs'):
    os.makedirs('../logs')

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
        # Define network structure: input 2 features (x,t), output 1 value (u)
        self.net = MLP([2, 64, 1])
        self.pi = torch.tensor(np.pi)

    def forward(self, x, t):
        # Combine inputs into tensor
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

    def f(self, x, t):
        """Calculate heat equation residual: ∂u/∂t - α∂²u/∂x² = 0"""
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = self.forward(x, t)

        # Calculate first derivative with respect to time
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate first derivative with respect to x
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate second derivative with respect to x
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # Heat equation: ∂u/∂t - α∂²u/∂x² = 0
        # Using α = 1 for simplicity
        residual = u_t - u_xx

        return residual


# Analytical solution
def exact_solution(x, t):
    return torch.sin(np.pi * x) * torch.exp(-np.pi ** 2 * t)


# Loss function calculation for APP
def loss_fun(model, params, inputs):
    # Save original parameters
    original_params = get_model_params(model).clone()

    # Set new parameters
    params_tensor = torch.tensor(params, dtype=torch.float64, device=device)
    set_model_params(model, params_tensor)

    # Calculate loss
    x_domain, t_domain, x_initial, t_initial, x_boundary, t_boundary = inputs

    # Calculate PDE residual loss
    f_pred = model.f(x_domain, t_domain)
    loss_f = torch.mean(torch.square(f_pred))

    # Calculate initial condition loss (u(x,0) = sin(πx))
    u_initial_pred = model(x_initial, t_initial)
    u_initial_true = torch.sin(model.pi * x_initial)
    loss_initial = torch.mean(torch.square(u_initial_pred - u_initial_true))

    # Calculate boundary condition loss (u(0,t) = u(1,t) = 0)
    u_boundary = model(x_boundary, t_boundary)
    loss_bc = torch.mean(torch.square(u_boundary))

    # Total loss
    loss = loss_f + loss_initial + loss_bc

    # Restore original parameters
    set_model_params(model, original_params)

    return loss.item()


# Train model using APP optimizer
def train_app(model, inputs, K, lambda_, rho, n):
    params = get_model_params(model)
    d = len(params)
    xk = params.detach().cpu().numpy().astype(np.float64)
    loss_history = []
    alpha = lambda_

    halton = qmc.Halton(d=d, scramble=True, seed=42)
    fc = np.inf
    for i in range(K):
        # Generate n random vectors from Halton sequence
        x = halton.random(n)
        t = np.vstack([xk, norm.ppf(x, loc=xk, scale=1 / alpha)])

        # Compute function value sequence
        f = [loss_fun(model, t[k], inputs) for k in range(n + 1)]
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


# Train model using L-BFGS optimizer
def train_lbfgs(model, inputs, epochs=200):
    # Unpack input data
    x_domain, t_domain, x_initial, t_initial, x_boundary, t_boundary = inputs
    
    # Create optimizer
    optimizer = optim.LBFGS(model.parameters(),
                            lr=1.0,
                            max_iter=20,
                            history_size=50,
                            tolerance_grad=1e-5,
                            tolerance_change=1e-7,
                            line_search_fn="strong_wolfe")
    
    # Training loop
    losses = []
    
    # Define closure function for L-BFGS
    def closure():
        # Zero gradients
        optimizer.zero_grad()
        
        # Calculate PDE residual loss
        f_pred = model.f(x_domain, t_domain)
        loss_f = torch.mean(torch.square(f_pred))

        # Calculate initial condition loss (u(x,0) = sin(πx))
        u_initial_pred = model(x_initial, t_initial)
        u_initial_true = torch.sin(model.pi * x_initial)
        loss_initial = torch.mean(torch.square(u_initial_pred - u_initial_true))

        # Calculate boundary condition loss (u(0,t) = u(1,t) = 0)
        u_boundary = model(x_boundary, t_boundary)
        loss_bc = torch.mean(torch.square(u_boundary))

        # Total loss
        loss = loss_f + loss_initial + loss_bc
        
        # Backward pass
        loss.backward()
        
        # Store current loss for printing
        closure.loss = loss.item()
        closure.loss_f = loss_f.item()
        closure.loss_initial = loss_initial.item()
        closure.loss_bc = loss_bc.item()
        
        return loss
    
    # Initialize loss values
    closure.loss = 0.0
    closure.loss_f = 0.0
    closure.loss_initial = 0.0
    closure.loss_bc = 0.0
    
    # Training loop
    for epoch in range(epochs):
        # Perform optimization step
        optimizer.step(closure)
        
        # Record loss
        losses.append(closure.loss)
        
        # Print training progress
        if epoch % 10 == 0:  # Print more frequently since L-BFGS converges faster
            print(f'L-BFGS - Epoch {epoch}, Loss: {closure.loss:.6e}, PDE Loss: {closure.loss_f:.6e}, IC Loss: {closure.loss_initial:.6e}, BC Loss: {closure.loss_bc:.6e}')

    return losses


# Evaluate and visualize results
def evaluate_model(model, n_points=100):
    # Create grid points
    x = np.linspace(0, 1, n_points)
    t = np.linspace(0, 1, n_points)
    X, T = np.meshgrid(x, t)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X.flatten()[:, None], device=device)
    T_tensor = torch.tensor(T.flatten()[:, None], device=device)

    # Predict
    model.eval()
    with torch.no_grad():
        u_pred = model(X_tensor, T_tensor).cpu().numpy()

    # Calculate analytical solution
    u_exact = exact_solution(X_tensor, T_tensor).cpu().numpy()

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
    axes[0].set_ylabel('t')
    plt.colorbar(im1, ax=axes[0])

    # Exact solution
    im2 = axes[1].imshow(U_exact, cmap='viridis', origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    axes[1].set_title('Exact Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    plt.colorbar(im2, ax=axes[1])

    # Error
    im3 = axes[2].imshow(Error, cmap='jet', origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig('./logs/heat_app_lbfgs_results.png', dpi=300)
    plt.show()

    # Calculate L2 relative error
    l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    print(f'L2 relative error: {l2_error:.6e}')

    # Plot solution at different time steps
    fig, ax = plt.subplots(figsize=(10, 6))
    time_steps = [0, 0.25, 0.5, 0.75, 1.0]
    for i, t in enumerate(time_steps):
        t_idx = int(t * (n_points - 1))
        ax.plot(x, U_pred[t_idx, :], '--', label=f'PINN t={t}')
        ax.plot(x, U_exact[t_idx, :], '-', label=f'Exact t={t}')

    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Solution at Different Time Steps')
    ax.legend()
    ax.grid(True)
    plt.savefig('./logs/heat_app_lbfgs_time_slices.png', dpi=300)
    plt.show()

    return U_pred, U_exact, Error, l2_error


# Main function
def main():
    # Create model
    model = PINN().to(device)

    # Domain boundaries
    x_min, x_max = 0.0, 1.0
    t_min, t_max = 0.0, 1.0
    n_points = 400

    # Interior points
    x_domain = torch.rand(n_points, 1, device=device) * (x_max - x_min) + x_min
    t_domain = torch.rand(n_points, 1, device=device) * (t_max - t_min) + t_min

    # Initial condition points (t=0)
    x_initial = torch.rand(n_points, 1, device=device) * (x_max - x_min) + x_min
    t_initial = torch.zeros(n_points, 1, device=device)

    # Boundary points (x=0, x=1)
    t_boundary = torch.rand(n_points, 1, device=device) * (t_max - t_min) + t_min
    x_boundary_0 = torch.zeros(n_points // 2, 1, device=device)
    x_boundary_1 = torch.ones(n_points // 2, 1, device=device)

    # Merge boundary points
    x_boundary = torch.cat([x_boundary_0, x_boundary_1])
    t_boundary = torch.cat([t_boundary[:n_points // 2], t_boundary[n_points // 2:]])

    # Prepare input data
    inputs = (x_domain, t_domain, x_initial, t_initial, x_boundary, t_boundary)

    # Step 1: Use APP optimization algorithm for initial training
    print("Step 1: APP optimization...")
    app_losses = train_app(
        model,
        inputs=inputs,
        K=400,
        lambda_=1 / np.sqrt(len(get_model_params(model))),
        rho=0.98,
        n=len(get_model_params(model))
    )

    # Step 2: Further training with L-BFGS optimizer
    print("Step 2: Further training with L-BFGS optimizer...")
    lbfgs_losses = train_lbfgs(model, inputs, epochs=200)

    # Plot combined loss curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(app_losses + lbfgs_losses)
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value (Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig('./logs/heat_app_lbfgs_loss.png', dpi=300)
    plt.show()

    # Evaluate model
    print("Evaluating model...")
    U_pred, U_exact, Error, l2_error = evaluate_model(model)


if __name__ == "__main__":
    main()