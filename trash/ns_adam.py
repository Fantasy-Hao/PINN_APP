import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sophia import SophiaG

# Create log directory
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


# Define PINN model for Lid-driven Cavity Flow
class PINN(nn.Module):
    def __init__(self, Re=100):
        super(PINN, self).__init__()
        # Define network structure: input 3 features (x,y,t), output 3 values (u,v,p)
        # u: x-direction velocity, v: y-direction velocity, p: pressure
        self.net = MLP([3, 64, 3])
        self.Re = Re  # Reynolds number

    def forward(self, x, y, t):
        # Combine inputs as tensor
        xyt = torch.cat([x, y, t], dim=1)
        output = self.net(xyt)
        
        # Separate outputs
        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]
        
        return u, v, p

    def f(self, x, y, t):
        """Calculate Navier-Stokes equation residuals"""
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        u, v, p = self.forward(x, y, t)

        # Calculate u derivatives
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

        u_y = torch.autograd.grad(
            u, y,
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

        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate v derivatives
        v_t = torch.autograd.grad(
            v, t,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True
        )[0]

        v_x = torch.autograd.grad(
            v, x,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True
        )[0]

        v_y = torch.autograd.grad(
            v, y,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True
        )[0]

        v_xx = torch.autograd.grad(
            v_x, x,
            grad_outputs=torch.ones_like(v_x),
            retain_graph=True,
            create_graph=True
        )[0]

        v_yy = torch.autograd.grad(
            v_y, y,
            grad_outputs=torch.ones_like(v_y),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calculate p derivatives
        p_x = torch.autograd.grad(
            p, x,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True
        )[0]

        p_y = torch.autograd.grad(
            p, y,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True
        )[0]

        # Incompressible Navier-Stokes equations
        # Continuity equation: ∂u/∂x + ∂v/∂y = 0
        continuity = u_x + v_y

        # Momentum equation x-direction: ∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + (1/Re)(∂²u/∂x² + ∂²u/∂y²)
        momentum_x = u_t + u * u_x + v * u_y + p_x - (1/self.Re) * (u_xx + u_yy)

        # Momentum equation y-direction: ∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + (1/Re)(∂²v/∂x² + ∂²v/∂y²)
        momentum_y = v_t + u * v_x + v * v_y + p_y - (1/self.Re) * (v_xx + v_yy)

        return continuity, momentum_x, momentum_y


# Generate training data
def generate_training_data(n_points=1000, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, t_min=0.0, t_max=1.0):
    # Interior points
    x_domain = torch.rand(n_points, 1, device=device) * (x_max - x_min) + x_min
    y_domain = torch.rand(n_points, 1, device=device) * (y_max - y_min) + y_min
    t_domain = torch.rand(n_points, 1, device=device) * (t_max - t_min) + t_min

    # Initial condition points (t=0)
    x_initial = torch.rand(n_points, 1, device=device) * (x_max - x_min) + x_min
    y_initial = torch.rand(n_points, 1, device=device) * (y_max - y_min) + y_min
    t_initial = torch.zeros(n_points, 1, device=device)

    # Boundary points
    # Bottom boundary (y=0)
    x_bottom = torch.rand(n_points//4, 1, device=device) * (x_max - x_min) + x_min
    y_bottom = torch.zeros(n_points//4, 1, device=device)
    t_bottom = torch.rand(n_points//4, 1, device=device) * (t_max - t_min) + t_min

    # Top boundary (y=1) - moving lid
    x_top = torch.rand(n_points//4, 1, device=device) * (x_max - x_min) + x_min
    y_top = torch.ones(n_points//4, 1, device=device)
    t_top = torch.rand(n_points//4, 1, device=device) * (t_max - t_min) + t_min

    # Left boundary (x=0)
    x_left = torch.zeros(n_points//4, 1, device=device)
    y_left = torch.rand(n_points//4, 1, device=device) * (y_max - y_min) + y_min
    t_left = torch.rand(n_points//4, 1, device=device) * (t_max - t_min) + t_min

    # Right boundary (x=1)
    x_right = torch.ones(n_points//4, 1, device=device)
    y_right = torch.rand(n_points//4, 1, device=device) * (y_max - y_min) + y_min
    t_right = torch.rand(n_points//4, 1, device=device) * (t_max - t_min) + t_min
    
    return (x_domain, y_domain, t_domain, 
            x_initial, y_initial, t_initial,
            x_bottom, y_bottom, t_bottom,
            x_top, y_top, t_top,
            x_left, y_left, t_left,
            x_right, y_right, t_right)


# Training function
def train(model, inputs, n_epochs):
    # Unpack input data
    (x_domain, y_domain, t_domain, 
     x_initial, y_initial, t_initial,
     x_bottom, y_bottom, t_bottom,
     x_top, y_top, t_top,
     x_left, y_left, t_left,
     x_right, y_right, t_right) = inputs

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
        continuity, momentum_x, momentum_y = model.f(x_domain, y_domain, t_domain)
        loss_pde = (torch.mean(torch.square(continuity)) + 
                    torch.mean(torch.square(momentum_x)) + 
                    torch.mean(torch.square(momentum_y)))

        # Calculate initial condition loss (fluid at rest when t=0)
        u_initial, v_initial, _ = model(x_initial, y_initial, t_initial)
        loss_initial = torch.mean(torch.square(u_initial)) + torch.mean(torch.square(v_initial))

        # Calculate boundary condition losses
        # Bottom boundary: u=0, v=0 (no-slip)
        u_bottom, v_bottom, _ = model(x_bottom, y_bottom, t_bottom)
        loss_bottom = torch.mean(torch.square(u_bottom)) + torch.mean(torch.square(v_bottom))

        # Top boundary: u=1, v=0 (moving lid)
        u_top, v_top, _ = model(x_top, y_top, t_top)
        loss_top = torch.mean(torch.square(u_top - 1.0)) + torch.mean(torch.square(v_top))

        # Left boundary: u=0, v=0 (no-slip)
        u_left, v_left, _ = model(x_left, y_left, t_left)
        loss_left = torch.mean(torch.square(u_left)) + torch.mean(torch.square(v_left))

        # Right boundary: u=0, v=0 (no-slip)
        u_right, v_right, _ = model(x_right, y_right, t_right)
        loss_right = torch.mean(torch.square(u_right)) + torch.mean(torch.square(v_right))

        # Total loss
        loss_bc = loss_bottom + loss_top + loss_left + loss_right
        loss = loss_pde + 10.0 * loss_initial + 10.0 * loss_bc

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
def evaluate_model(model, t_value=1.0, n_points=100):
    # Create grid points
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    T = np.ones_like(X) * t_value

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X.flatten()[:, None], device=device)
    Y_tensor = torch.tensor(Y.flatten()[:, None], device=device)
    T_tensor = torch.tensor(T.flatten()[:, None], device=device)

    # Predict
    model.eval()
    with torch.no_grad():
        u_pred, v_pred, p_pred = model(X_tensor, Y_tensor, T_tensor)
        u_pred = u_pred.cpu().numpy()
        v_pred = v_pred.cpu().numpy()
        p_pred = p_pred.cpu().numpy()

    # Reshape to grid shape
    U_pred = u_pred.reshape(n_points, n_points)
    V_pred = v_pred.reshape(n_points, n_points)
    P_pred = p_pred.reshape(n_points, n_points)

    # Calculate velocity normalization for streamlines
    velocity_norm = np.linalg.norm(np.sqrt(U_pred**2 + V_pred**2).flatten(), 2)
    print(f'Velocity field L2 norm: {velocity_norm:.6e}')

    # Calculate velocity magnitude for streamlines
    velocity_magnitude = np.sqrt(U_pred**2 + V_pred**2)

    # Visualization - only results figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Horizontal velocity component (u)
    im1 = axes[0, 0].imshow(U_pred, cmap='viridis', origin='lower', extent=[0, 1, 0, 1], aspect='equal')
    axes[0, 0].set_title('Horizontal Velocity (u)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])

    # Vertical velocity component (v)
    im2 = axes[0, 1].imshow(V_pred, cmap='viridis', origin='lower', extent=[0, 1, 0, 1], aspect='equal')
    axes[0, 1].set_title('Vertical Velocity (v)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])

    # Pressure field (p)
    im3 = axes[1, 0].imshow(P_pred, cmap='viridis', origin='lower', extent=[0, 1, 0, 1], aspect='equal')
    axes[1, 0].set_title('Pressure Field (p)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])

    # Velocity streamlines
    axes[1, 1].streamplot(X, Y, U_pred, V_pred, density=1.0, color=velocity_magnitude, cmap='viridis')
    axes[1, 1].set_title('Velocity Streamlines')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('./logs/ns_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return U_pred, V_pred, P_pred

# Main function
def main():
    # Create model
    model = PINN(Re=100).to(device)

    # Prepare input data
    inputs = generate_training_data(n_points=2000)

    # Train model
    print("Starting training...")
    losses = train(model, inputs, n_epochs=20400)

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value (log scale)')
    plt.grid(True)
    plt.savefig('./logs/ns_loss.png', dpi=300)
    plt.show()

    # Evaluate model
    print("Evaluating model...")
    U_pred, V_pred, P_pred = evaluate_model(model, t_value=1.0)


if __name__ == "__main__":
    main()