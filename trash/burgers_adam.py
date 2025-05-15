import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
from pyDOE import lhs

from sophia import SophiaG

# Create logs directory
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


# Define PINN model for 1D Burgers' equation
class PINN(nn.Module):
    def __init__(self, nu=0.01/np.pi):
        super(PINN, self).__init__()
        # Define network structure: input 2 features (x,t), output 1 value (u)
        self.net = MLP([2, 20, 20, 20, 20, 20, 20, 20, 20, 1])
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


# Load and prepare training data
def load_and_prepare_data():
    # Set parameters
    N_u = 100
    N_f = 10000
    
    # Load data
    data = scipy.io.loadmat('./data/burgers_shock.mat')
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x, t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              
    
    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
    
    # Prepare training data
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]
    
    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    
    # Randomly select boundary points
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]
    
    # Convert to PyTorch tensors
    x_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).to(device)
    t_u = torch.tensor(X_u_train[:, 1:2], requires_grad=True).to(device)
    u = torch.tensor(u_train).to(device)
    x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).to(device)
    t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).to(device)
    
    return x_u, t_u, u, x_f, t_f, X_star, u_star, lb, ub


# Training function
def train(model, inputs, n_epochs):
    # Unpack input data
    x_u, t_u, u, x_f, t_f = inputs

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # optimizer = optim.NAdam(model.parameters(), lr=1e-4)
    # optimizer = optim.RAdam(model.parameters(), lr=1e-4)
    # optimizer = SophiaG(model.parameters(), lr=1e-4)

    # Training loop
    losses = []
    for epoch in range(n_epochs):
        # Calculate PDE residual loss
        f_pred = model.f(x_f, t_f)
        loss_f = torch.mean(torch.square(f_pred))

        # Calculate data loss
        u_pred = model(x_u, t_u)
        loss_u = torch.mean(torch.square(u - u_pred))

        # Total loss
        loss = loss_u + loss_f

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
def evaluate_model(model, X_star, u_star, lb, ub):
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_star[:, 0:1], device=device)
    T_tensor = torch.tensor(X_star[:, 1:2], device=device)

    # Predict
    model.eval()
    with torch.no_grad():
        u_pred = model(X_tensor, T_tensor)
        u_pred = u_pred.cpu().numpy()

    # Calculate error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print(f'Relative L2 Error: {error_u:.6e}')

    # Visualization
    # Rebuild grid
    x = np.linspace(lb[0], ub[0], 100)
    t = np.linspace(lb[1], ub[1], 100)
    X, T = np.meshgrid(x, t)
    
    # Predict over entire domain
    X_star_vis = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    X_tensor_vis = torch.tensor(X_star_vis[:, 0:1], device=device)
    T_tensor_vis = torch.tensor(X_star_vis[:, 1:2], device=device)
    
    with torch.no_grad():
        u_pred_vis = model(X_tensor_vis, T_tensor_vis)
        u_pred_vis = u_pred_vis.cpu().numpy()
    
    U_pred = u_pred_vis.reshape(X.shape)
    
    # Load exact solution data
    data = scipy.io.loadmat('./data/burgers_shock.mat')
    t_exact = data['t'].flatten()
    x_exact = data['x'].flatten()
    Exact = np.real(data['usol'])
    
    # Interpolate exact solution to match prediction grid if needed
    from scipy.interpolate import griddata
    X_exact, T_exact = np.meshgrid(x_exact, t_exact)
    points = np.hstack((X_exact.flatten()[:, None], T_exact.flatten()[:, None]))
    values = Exact.flatten()
    Exact_grid = griddata(points, values, (X, T), method='cubic')
    Exact_grid = Exact_grid.reshape(X.shape)
    
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Predicted solution
    cf1 = axs[0].contourf(X, T, U_pred, 100, cmap='viridis')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('t')
    axs[0].set_title('Predicted Solution')
    fig.colorbar(cf1, ax=axs[0])
    
    # 2. Absolute error
    error = np.abs(Exact_grid - U_pred)
    cf2 = axs[1].contourf(X, T, error, 100, cmap='jet')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('t')
    axs[1].set_title('Absolute Error')
    fig.colorbar(cf2, ax=axs[1])
    
    plt.tight_layout()
    plt.savefig('./logs/burgers_results.png', dpi=300)
    plt.show()

    return U_pred


# Main function
def main():
    # Load data
    x_u, t_u, u, x_f, t_f, X_star, u_star, lb, ub = load_and_prepare_data()
    
    # Create model
    nu = 0.01/np.pi  # Viscosity coefficient
    model = PINN(nu=nu).to(device)

    # Prepare input data
    inputs = (x_u, t_u, u, x_f, t_f)

    # Train model
    print("Starting training...")
    losses = train(model, inputs, n_epochs=20000)

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value (log scale)')
    plt.grid(True)
    plt.savefig('./logs/burgers_loss.png', dpi=300)
    plt.show()

    # Evaluate model
    print("Evaluating model...")
    U_pred = evaluate_model(model, X_star, u_star, lb, ub)


if __name__ == "__main__":
    main()