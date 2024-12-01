#Jam Sadiq Oct 25, 2024
#Kepler circular orbit with PINNS
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np

# Define the neural network
class PINN_Orbit(nn.Module):
    def __init__(self):
        super(PINN_Orbit, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2)  # Output: [x(t), y(t)]
        )

    def forward(self, t):
        return self.hidden(t)

# Physics-Informed Loss function
def physics_loss(model, t, G, M):
    t.requires_grad = True
    output = model(t)
    x = output[:, 0:1]
    y = output[:, 1:2]
    
    # First derivatives
    x_t = autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    y_t = autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
    
    # Second derivatives
    x_tt = autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]
    y_tt = autograd.grad(y_t, t, torch.ones_like(y_t), create_graph=True)[0]
    
    # Radial distance
    r = torch.sqrt(x**2 + y**2)
    
    # Newtonian equations of motion
    fx = x_tt + (G * M * x) / (r**3)
    fy = y_tt + (G * M * y) / (r**3)
    
    # Residual loss
    return torch.mean(fx**2 + fy**2)

# Initial condition loss
def initial_condition_loss(model, t0, x0, y0, vx0, vy0):
    output = model(t0)
    x_pred = output[:, 0:1]
    y_pred = output[:, 1:2]
    
    # First derivatives
    x_t_pred = autograd.grad(x_pred, t0, torch.ones_like(x_pred), create_graph=True)[0]
    y_t_pred = autograd.grad(y_pred, t0, torch.ones_like(y_pred), create_graph=True)[0]
    
    # Initial condition losses
    loss_x0 = torch.mean((x_pred - x0)**2)
    loss_y0 = torch.mean((y_pred - y0)**2)
    loss_vx0 = torch.mean((x_t_pred - vx0)**2)
    loss_vy0 = torch.mean((y_t_pred - vy0)**2)
    
    return loss_x0 + loss_y0 + loss_vx0 + loss_vy0

# Training the PINN
def train_pinn(G, M, num_epochs=5000, lr=1e-3):
    model = PINN_Orbit()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Time domain for training
    t = torch.linspace(0, 2*np.pi, 100).view(-1, 1)  # One orbit (in radians)
    t0 = torch.tensor([[0.0]], requires_grad=True)
    
    # Initial conditions for a circular orbit
    x0 = torch.tensor([[1.0]])  # Unit distance
    y0 = torch.tensor([[0.0]])  # Start on x-axis
    vx0 = torch.tensor([[0.0]])  # Perpendicular velocity
    vy0 = torch.tensor([[1.0]])  # Orbital velocity (G*M/r = 1 for unit system)
    
    loss_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Loss from physics-informed residual
        p_loss = physics_loss(model, t, G, M)

        # Loss from initial conditions
        ic_loss = initial_condition_loss(model, t0, x0, y0, vx0, vy0)

        # Total loss
        loss = p_loss + ic_loss
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model, loss_history

# Define gravitational constant and mass
G = 1.0  # Gravitational constant (unit system)
M = 1.0  # Mass of the central body (unit system)

# Train the PINN
model, loss_history = train_pinn(G, M)

# Plotting the orbit
t_test = torch.linspace(0, 2*np.pi, 200).view(-1, 1)
orbit_pred = model(t_test).detach().numpy()

# Extract x and y components
x_pred = orbit_pred[:, 0]
y_pred = orbit_pred[:, 1]

# Analytical solution for comparison
x_exact = np.cos(t_test.numpy())
y_exact = np.sin(t_test.numpy())

plt.figure(figsize=(8, 8))
plt.plot(x_pred, y_pred, label="PINN Solution")
plt.plot(x_exact, y_exact, label="Exact Solution", linestyle="dashed")
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.title("Orbit: PINN vs Analytical Solution")
plt.legend()
plt.axis("equal")
plt.show()

# Plot loss history
plt.figure()
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss History")
plt.show()
