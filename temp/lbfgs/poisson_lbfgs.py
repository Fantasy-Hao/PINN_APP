"""
LBFGS Optimization.
Two dimensional Poisson equation example. Solution given by

u(x,y) = sin(pi*x) * sin(pi*y).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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


# Training function
def train(model, optimizer, max_iter=1000, n_points=100):
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

    # Training loop
    losses = []
    iteration = 0
    
    # Define closure function for LBFGS
    def closure():
        nonlocal iteration
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Calculate PDE residual loss
        f_pred = model.f(x_domain, y_domain)
        loss_f = torch.mean(torch.square(f_pred))
        
        # Calculate boundary condition loss (Dirichlet boundary condition: u=0)
        u_boundary = model(x_boundary, y_boundary)
        loss_bc = torch.mean(torch.square(u_boundary))
        
        # Total loss
        loss = loss_f + loss_bc
        
        # Backward pass
        loss.backward()
        
        # Print training progress
        if iteration % 100 == 0:
            print(
                f'LBFGS - Iteration {iteration}, Loss: {loss.item():.6e}, PDE Loss: {loss_f.item():.6e}, BC Loss: {loss_bc.item():.6e}')
        
        # Record loss
        losses.append(loss.item())
        
        # Increment iteration counter
        iteration += 1
        
        return loss
    
    # Run optimization
    for i in range(max_iter):
        optimizer.step(closure)
    
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
    plt.savefig('./logs/poisson_lbfgs_results.png', dpi=300)
    plt.show()

    # Calculate L2 relative error
    l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    print(f'L2 relative error: {l2_error:.6e}')

    return U_pred, U_exact, Error, l2_error


# Main function
def main():
    # Create model
    model = PINN().to(device)

    # Define optimizer - LBFGS
    optimizer = optim.LBFGS(model.parameters(),
                            lr=1.0,
                            max_iter=20,
                            history_size=50,
                            tolerance_grad=1e-5,
                            tolerance_change=1e-7,
                            line_search_fn="strong_wolfe")

    # Train model
    print("Starting training...")
    losses = train(model, optimizer, max_iter=2000, n_points=800)

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value (Log Scale)')
    plt.grid(True)
    plt.savefig('./logs/poisson_lbfgs_loss.png', dpi=300)
    plt.show()

    # Evaluate model
    print("Evaluating model...")
    U_pred, U_exact, Error, l2_error = evaluate_model(model)

    # Save model
    # torch.save(model.state_dict(), 'poisson_lbfgs_model.pt')
    # print("Model saved as 'poisson_lbfgs_model.pt'")


if __name__ == "__main__":
    main()