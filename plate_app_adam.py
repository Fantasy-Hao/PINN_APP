import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc, norm

from sophia import SophiaG
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


# Define PINN model for plate bending problem
class PINN(nn.Module):
    def __init__(self, D=1.0, mu=0.28):
        super(PINN, self).__init__()
        # Define network structure: input 2 features (x,y), output 1 value (w)
        self.net = MLP([2, 32, 1])
        self.D = D  # Plate bending stiffness
        self.mu = mu  # Poisson's ratio

    def forward(self, x, y):
        # Combine inputs into tensor
        xy = torch.cat([x, y], dim=1)
        return self.net(xy)

    def f(self, x, y, q=1000.0):
        """Calculate biharmonic equation residual: ∇⁴w = q/D"""
        x.requires_grad_(True)
        y.requires_grad_(True)

        w = self.forward(x, y)

        # Calculate first derivative with respect to x
        w_x = torch.autograd.grad(
            w, x,
            grad_outputs=torch.ones_like(w),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate second derivative with respect to x
        w_xx = torch.autograd.grad(
            w_x, x,
            grad_outputs=torch.ones_like(w_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate first derivative with respect to y
        w_y = torch.autograd.grad(
            w, y,
            grad_outputs=torch.ones_like(w),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate second derivative with respect to y
        w_yy = torch.autograd.grad(
            w_y, y,
            grad_outputs=torch.ones_like(w_y),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate mixed second derivative
        w_xy = torch.autograd.grad(
            w_x, y,
            grad_outputs=torch.ones_like(w_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate third derivatives
        w_xxx = torch.autograd.grad(
            w_xx, x,
            grad_outputs=torch.ones_like(w_xx),
            retain_graph=True,
            create_graph=True
        )[0]

        w_xxy = torch.autograd.grad(
            w_xx, y,
            grad_outputs=torch.ones_like(w_xx),
            retain_graph=True,
            create_graph=True
        )[0]

        w_yyy = torch.autograd.grad(
            w_yy, y,
            grad_outputs=torch.ones_like(w_yy),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate fourth derivatives
        w_xxxx = torch.autograd.grad(
            w_xxx, x,
            grad_outputs=torch.ones_like(w_xxx),
            retain_graph=True,
            create_graph=True
        )[0]

        w_xxyy = torch.autograd.grad(
            w_xxy, y,
            grad_outputs=torch.ones_like(w_xxy),
            retain_graph=True,
            create_graph=True
        )[0]

        w_yyyy = torch.autograd.grad(
            w_yyy, y,
            grad_outputs=torch.ones_like(w_yyy),
            retain_graph=True,
            create_graph=True
        )[0]

        # Biharmonic equation: ∇⁴w = w_xxxx + 2*w_xxyy + w_yyyy = q/D
        residual = w_xxxx + 2*w_xxyy + w_yyyy - q/self.D

        return residual
    
    def shear_force(self, x, y):
        """Calculate shear force: ∂²w/∂y² + μ∂²w/∂x²"""
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        w = self.forward(x, y)
        
        # Calculate second derivatives
        w_x = torch.autograd.grad(
            w, x,
            grad_outputs=torch.ones_like(w),
            retain_graph=True,
            create_graph=True
        )[0]
        
        w_xx = torch.autograd.grad(
            w_x, x,
            grad_outputs=torch.ones_like(w_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        w_y = torch.autograd.grad(
            w, y,
            grad_outputs=torch.ones_like(w),
            retain_graph=True,
            create_graph=True
        )[0]
        
        w_yy = torch.autograd.grad(
            w_y, y,
            grad_outputs=torch.ones_like(w_y),
            retain_graph=True,
            create_graph=True
        )[0]
        
        return w_yy + self.mu * w_xx
    
    def moment(self, x, y):
        """Calculate moment: ∂³w/∂y³ + (2-μ)∂³w/∂x²∂y"""
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        w = self.forward(x, y)
        
        # Calculate derivatives
        w_x = torch.autograd.grad(
            w, x,
            grad_outputs=torch.ones_like(w),
            retain_graph=True,
            create_graph=True
        )[0]
        
        w_xx = torch.autograd.grad(
            w_x, x,
            grad_outputs=torch.ones_like(w_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        w_xxy = torch.autograd.grad(
            w_xx, y,
            grad_outputs=torch.ones_like(w_xx),
            retain_graph=True,
            create_graph=True
        )[0]
        
        w_y = torch.autograd.grad(
            w, y,
            grad_outputs=torch.ones_like(w),
            retain_graph=True,
            create_graph=True
        )[0]
        
        w_yy = torch.autograd.grad(
            w_y, y,
            grad_outputs=torch.ones_like(w_y),
            retain_graph=True,
            create_graph=True
        )[0]
        
        w_yyy = torch.autograd.grad(
            w_yy, y,
            grad_outputs=torch.ones_like(w_yy),
            retain_graph=True,
            create_graph=True
        )[0]
        
        return w_yyy + (2 - self.mu) * w_xxy


# Generate training data
def generate_training_data(n_points, Lx=2.0, Ly=1.0):
    """
    Generate training data for plate bending problem
    Lx, Ly: plate dimensions
    """
    # Interior points in domain [-Lx/2, Lx/2] x [-Ly/2, Ly/2]
    x_domain = (torch.rand(n_points, 1, device=device) - 0.5) * Lx
    y_domain = (torch.rand(n_points, 1, device=device) - 0.5) * Ly

    # Boundary points
    n_boundary = n_points // 4
    
    # Left boundary (x = -Lx/2)
    x_left = torch.full((n_boundary, 1), -Lx/2, device=device)
    y_left = (torch.rand(n_boundary, 1, device=device) - 0.5) * Ly
    
    # Right boundary (x = Lx/2)
    x_right = torch.full((n_boundary, 1), Lx/2, device=device)
    y_right = (torch.rand(n_boundary, 1, device=device) - 0.5) * Ly
    
    # Bottom boundary (y = -Ly/2)
    x_bottom = (torch.rand(n_boundary, 1, device=device) - 0.5) * Lx
    y_bottom = torch.full((n_boundary, 1), -Ly/2, device=device)
    
    # Top boundary (y = Ly/2)
    x_top = (torch.rand(n_boundary, 1, device=device) - 0.5) * Lx
    y_top = torch.full((n_boundary, 1), Ly/2, device=device)
    
    # Combine boundary points
    x_boundary_lr = torch.cat([x_left, x_right])  # Left-right boundaries
    y_boundary_lr = torch.cat([y_left, y_right])
    
    x_boundary_tb = torch.cat([x_bottom, x_top])  # Top-bottom boundaries
    y_boundary_tb = torch.cat([y_bottom, y_top])
    
    return x_domain, y_domain, x_boundary_lr, y_boundary_lr, x_boundary_tb, y_boundary_tb


def loss_fun(model, params, inputs, q=1000.0):
    """Calculate loss for APP optimizer"""
    # Unpack input data
    x_domain, y_domain, x_boundary_lr, y_boundary_lr, x_boundary_tb, y_boundary_tb = inputs
    
    # Set model parameters
    set_model_params(model, torch.tensor(params, dtype=torch.float64, device=device))
    
    # Calculate PDE residual loss
    f_pred = model.f(x_domain, y_domain, q)
    loss_pde = torch.mean(torch.square(f_pred))
    
    # Left-right boundary conditions (simply supported: w=0, ∂²w/∂x²=0)
    w_lr = model(x_boundary_lr, y_boundary_lr)
    
    # Calculate ∂²w/∂x² for left-right boundaries
    x_boundary_lr.requires_grad_(True)
    y_boundary_lr.requires_grad_(True)
    w_lr_grad = model(x_boundary_lr, y_boundary_lr)
    w_x = torch.autograd.grad(
        w_lr_grad, x_boundary_lr,
        grad_outputs=torch.ones_like(w_lr_grad),
        retain_graph=True,
        create_graph=True
    )[0]
    w_xx = torch.autograd.grad(
        w_x, x_boundary_lr,
        grad_outputs=torch.ones_like(w_x),
        retain_graph=True,
        create_graph=True
    )[0]
    
    loss_bc_lr = torch.mean(torch.square(w_lr)) + torch.mean(torch.square(w_xx))
    
    # Top-bottom boundary conditions (free edge: shear force=0, moment=0)
    shear_tb = model.shear_force(x_boundary_tb, y_boundary_tb)
    moment_tb = model.moment(x_boundary_tb, y_boundary_tb)
    loss_bc_tb = torch.mean(torch.square(shear_tb)) + torch.mean(torch.square(moment_tb))
    
    # Total loss
    loss = loss_pde + 10.0 * loss_bc_lr + 10.0 * loss_bc_tb
    
    return loss.item()


# Train model using APP optimizer
def train_app(model, inputs, K, lambda_, rho, n, q=1000.0):
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
        f = [loss_fun(model, t[k], inputs, q) for k in range(n + 1)]
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


# Training function using Adam optimizer
def train_adam(model, inputs, n_epochs, q=1000.0):
    # Unpack input data
    x_domain, y_domain, x_boundary_lr, y_boundary_lr, x_boundary_tb, y_boundary_tb = inputs

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
        f_pred = model.f(x_domain, y_domain, q)
        loss_pde = torch.mean(torch.square(f_pred))

        # Left-right boundary conditions (simply supported: w=0, ∂²w/∂x²=0)
        w_lr = model(x_boundary_lr, y_boundary_lr)
        
        # Calculate ∂²w/∂x² for left-right boundaries
        x_boundary_lr.requires_grad_(True)
        y_boundary_lr.requires_grad_(True)
        w_lr_grad = model(x_boundary_lr, y_boundary_lr)
        w_x = torch.autograd.grad(
            w_lr_grad, x_boundary_lr,
            grad_outputs=torch.ones_like(w_lr_grad),
            retain_graph=True,
            create_graph=True
        )[0]
        w_xx = torch.autograd.grad(
            w_x, x_boundary_lr,
            grad_outputs=torch.ones_like(w_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        loss_bc_lr = torch.mean(torch.square(w_lr)) + torch.mean(torch.square(w_xx))

        # Top-bottom boundary conditions (free edge: shear force=0, moment=0)
        shear_tb = model.shear_force(x_boundary_tb, y_boundary_tb)
        moment_tb = model.moment(x_boundary_tb, y_boundary_tb)
        loss_bc_tb = torch.mean(torch.square(shear_tb)) + torch.mean(torch.square(moment_tb))

        # Total loss
        loss = loss_pde + 10.0 * loss_bc_lr + 10.0 * loss_bc_tb

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
def evaluate_model(model, n_points=50, Lx=2.0, Ly=1.0, q=1000.0, D=1.0):
    # Create grid points
    x = np.linspace(-Lx/2, Lx/2, n_points)
    y = np.linspace(-Ly/2, Ly/2, n_points)
    X, Y = np.meshgrid(x, y)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X.flatten()[:, None], device=device)
    Y_tensor = torch.tensor(Y.flatten()[:, None], device=device)

    # Predict
    model.eval()
    with torch.no_grad():
        w_pred = model(X_tensor, Y_tensor).cpu().numpy()

    # Reshape to grid shape
    W_pred = w_pred.reshape(n_points, n_points)

    # Visualization - 2D contour plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Predicted solution
    im = ax.contourf(X, Y, W_pred, cmap='viridis', levels=20)
    ax.set_title('PINN Predicted Solution - Plate Deflection')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.colorbar(im, ax=ax, label='Deflection w (m)')

    plt.tight_layout()
    plt.savefig('./logs/plate_bending_results.png', dpi=300)
    plt.show()

    # Print maximum deflection
    max_deflection = np.max(np.abs(W_pred))
    print(f'Maximum deflection: {max_deflection:.6e} m')

    return W_pred


# Main function
def main():
    # Set parameters according to user input
    Lx = 2.0  # plate length
    Ly = 1.0  # plate width
    q = 1000.0  # uniform load (N/m²)
    E = 210000.0e6  # Young's modulus (Pa)
    h = 0.01  # plate thickness (m)
    mu = 0.28  # Poisson's ratio
    
    # Calculate bending stiffness
    D = E * (h**3) / (12 * (1 - mu**2))
    print(f'Bending stiffness D = {D:.2e} N⋅m')
    
    # Create model
    model = PINN(D=D, mu=mu).to(device)

    # Generate training data
    n_points = 2000
    inputs = generate_training_data(n_points, Lx, Ly)

    # Step 1: Use APP optimization algorithm
    print("Step 1: APP optimization...")
    app_losses = train_app(
        model,
        inputs=inputs,
        K=400,
        lambda_=1 / np.sqrt(len(get_model_params(model))),
        rho=0.98,
        n=len(get_model_params(model)),
        q=q
    )

    # Step 2: Further training with Adam optimizer
    print("Step 2: Further training with Adam optimizer...")
    adam_losses = train_adam(
        model,
        inputs=inputs,
        n_epochs=20000,
        q=q
    )
    
    # Save losses to txt file
    all_losses = app_losses + adam_losses
    with open('./logs/plate_app_losses.txt', 'w') as f:
        for i, loss in enumerate(all_losses):
            f.write(f'{i}\t{loss:.6e}\n')
    print("Losses saved to ./logs/plate_app_losses.txt")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(app_losses + adam_losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.grid(True)
    plt.savefig('./logs/plate_app_bending_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Evaluate model
    print("Evaluating model...")
    W_pred = evaluate_model(model, n_points=50, Lx=Lx, Ly=Ly, q=q, D=D)


if __name__ == "__main__":
    main()