import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc, norm

from sophia import SophiaG
from utils import get_model_params, set_model_params

# Create log directory
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


# Define PINN model for 1D Burgers' equation
class PINN(nn.Module):
    def __init__(self, nu=0.01/np.pi):
        super(PINN, self).__init__()
        # Define network structure: input 2 features (x,t), output 1 value (u)
        self.net = MLP([2, 64, 1])
        self.nu = nu  # Viscosity coefficient

    def forward(self, x, t):
        # Combine inputs as tensor
        xt = torch.cat([x, t], dim=1)
        u = self.net(xt)
        return u

    def f(self, x, t):
        """Calculate Burgers' equation residual"""
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = self.forward(x, t)

        # Calculate derivatives
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

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

        # Burgers' equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
        residual = u_t + u * u_x - self.nu * u_xx

        return residual


# Generate training data
def generate_training_data(n_domain=800, n_boundary=100, n_initial=100, 
                           x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0):
    # Domain points
    x_domain = torch.rand(n_domain, 1, device=device) * (x_max - x_min) + x_min
    t_domain = torch.rand(n_domain, 1, device=device) * (t_max - t_min) + t_min

    # Boundary points (x=-1 and x=1)
    x_boundary_left = torch.ones(n_boundary, 1, device=device) * x_min
    x_boundary_right = torch.ones(n_boundary, 1, device=device) * x_max
    t_boundary = torch.rand(n_boundary, 1, device=device) * (t_max - t_min) + t_min

    # Initial condition points (t=0)
    x_initial = torch.rand(n_initial, 1, device=device) * (x_max - x_min) + x_min
    t_initial = torch.zeros(n_initial, 1, device=device)
    
    return (x_domain, t_domain, 
            x_boundary_left, x_boundary_right, t_boundary,
            x_initial, t_initial)


# Initial condition function: u(x,0) = -sin(πx)
def initial_condition(x):
    return -torch.sin(np.pi * x)


# Boundary conditions: u(-1,t) = u(1,t) = 0
def boundary_condition(x):
    return torch.zeros_like(x)


# Loss function for APP optimizer
def loss_fun(model, params, inputs):
    """Calculate loss for APP optimizer"""
    # Unpack input data
    (x_domain, t_domain, 
     x_boundary_left, x_boundary_right, t_boundary,
     x_initial, t_initial) = inputs
    
    # Set model parameters
    set_model_params(model, torch.tensor(params, dtype=torch.float64, device=device))
    
    # Calculate PDE residual loss
    residual = model.f(x_domain, t_domain)
    loss_pde = torch.mean(torch.square(residual))

    # Calculate boundary condition loss
    u_boundary_left = model(x_boundary_left, t_boundary)
    u_boundary_right = model(x_boundary_right, t_boundary)
    loss_bc = (torch.mean(torch.square(u_boundary_left)) + 
               torch.mean(torch.square(u_boundary_right)))

    # Calculate initial condition loss
    u_initial_pred = model(x_initial, t_initial)
    u_initial_true = initial_condition(x_initial)
    loss_ic = torch.mean(torch.square(u_initial_pred - u_initial_true))

    # Total loss
    loss = loss_pde + 10.0 * loss_bc + 10.0 * loss_ic
    
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

        # Calculate function value sequence
        f = [loss_fun(model, t[k], inputs) for k in range(n + 1)]
        fk = f[0]
        f_min = min(f)
        fc = min(fc, f_min)
        f = np.array(f) - f_min

        # Use average asymptotic formula
        f_mean = np.mean(f)
        if f_mean > 0:
            f /= f_mean

        # Calculate weights and new xk
        weights = np.exp(-f)
        xk = np.average(t, axis=0, weights=weights)

        # Update parameters and record
        set_model_params(model, torch.tensor(xk, dtype=torch.float64, device=device))
        alpha /= rho
        loss_history.append(fk)
        print(f'APP - Epoch {i + 1}, Loss: {fk:.6e}')

    return loss_history


# Train function using Adam
def train_adam(model, inputs, n_epochs):
    # Unpack input data
    (x_domain, t_domain, 
     x_boundary_left, x_boundary_right, t_boundary,
     x_initial, t_initial) = inputs

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
        residual = model.f(x_domain, t_domain)
        loss_pde = torch.mean(torch.square(residual))

        # Calculate boundary condition loss
        u_boundary_left = model(x_boundary_left, t_boundary)
        u_boundary_right = model(x_boundary_right, t_boundary)
        loss_bc = (torch.mean(torch.square(u_boundary_left)) + 
                   torch.mean(torch.square(u_boundary_right)))

        # Calculate initial condition loss
        u_initial_pred = model(x_initial, t_initial)
        u_initial_true = initial_condition(x_initial)
        loss_ic = torch.mean(torch.square(u_initial_pred - u_initial_true))

        # Total loss
        loss = loss_pde + 10.0 * loss_bc + 10.0 * loss_ic

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
def evaluate_model(model, n_x=100, n_t=100, x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0):
    # Create grid points
    x = np.linspace(x_min, x_max, n_x)
    t = np.linspace(t_min, t_max, n_t)
    X, T = np.meshgrid(x, t)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X.flatten()[:, None], device=device)
    T_tensor = torch.tensor(T.flatten()[:, None], device=device)

    # Predict
    model.eval()
    with torch.no_grad():
        u_pred = model(X_tensor, T_tensor)
        u_pred = u_pred.cpu().numpy()

    # Reshape to grid shape
    U_pred = u_pred.reshape(n_t, n_x)

    # Visualization - Surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, U_pred, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    ax.set_title("Solution of Burgers' Equation")
    fig.colorbar(surf)
    plt.savefig('./logs/burgers_app_solution_3d.png', dpi=300)
    plt.show()

    # Visualization - 2D plots at different time steps
    plt.figure(figsize=(12, 8))
    time_steps = [0, 25, 50, 75, 99]  # Indices for different time steps
    for i, t_idx in enumerate(time_steps):
        plt.subplot(2, 3, i+1)
        plt.plot(x, U_pred[t_idx, :])
        plt.title(f't = {t[t_idx]:.2f}')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('./logs/burgers_app_solution_2d.png', dpi=300)
    plt.show()

    # Calculate error
    error_u = np.linalg.norm(U_pred.flatten(), 2)
    print(f'Solution L2 norm: {error_u:.6e}')

    return U_pred


# Main function
def main():
    # Create model
    nu = 0.01/np.pi  # Viscosity coefficient
    model = PINN(nu=nu).to(device)

    # Prepare input data
    inputs = generate_training_data(n_domain=800, n_boundary=100, n_initial=100)

    # Step 1: Use APP optimization algorithm
    print("Step 1: APP optimization...")
    app_losses = train_app(
        model,
        inputs=inputs,
        K=400,
        lambda_=1 / np.sqrt(len(get_model_params(model))),
        rho=0.98,
        n=len(get_model_params(model))
    )

    # Step 2: Further training with Adam optimizer
    print("Step 2: Further training with Adam optimizer...")
    adam_losses = train_adam(
        model,
        inputs=inputs,
        n_epochs=20000
    )

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(app_losses + adam_losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value (log scale)')
    plt.grid(True)
    plt.savefig('./logs/burgers_app_loss.png', dpi=300)
    plt.show()

    # Evaluate model
    print("Evaluating model...")
    n_x, n_t = 100, 100
    x_min, x_max = -1.0, 1.0
    t_min, t_max = 0.0, 1.0
    
    # PINN solution
    U_pred = evaluate_model(model, n_x=n_x, n_t=n_t, 
                           x_min=x_min, x_max=x_max, 
                           t_min=t_min, t_max=t_max)


if __name__ == "__main__":
    main()