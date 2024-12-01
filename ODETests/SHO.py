#Jam Sadiq, Oct 8, 2024
#first try on PINNS for SHO problem
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np

# Define the neural network for approximating the solution
class PINN_SHO(nn.Module):
    def __init__(self):
        super(PINN_SHO, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.hidden(t)

# Physics-Informed Loss function
def physics_loss(model, t, omega):
    t.requires_grad = True
    u = model(t)
    
    # First derivative
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    
    # Second derivative
    u_tt = autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    
    # Residual of the SHO ODE
    residual = u_tt + omega**2 * u
    return torch.mean(residual**2)

# Initial condition loss
def initial_condition_loss(model, t0, u0, v0):
    u_pred = model(t0)
    u_t_pred = autograd.grad(u_pred, t0, torch.ones_like(u_pred), create_graph=True)[0]
    
    loss_u0 = torch.mean((u_pred - u0)**2)  # u(0) = u0
    loss_v0 = torch.mean((u_t_pred - v0)**2)  # du/dt(0) = v0
    return loss_u0 + loss_v0

# Training the PINN
def train_pinn(omega, num_epochs=5000, lr=1e-3):
    model = PINN_SHO()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Time domain for training
    t = torch.linspace(0, 10, 100).view(-1, 1)
    t0 = torch.tensor([[0.0]], requires_grad=True)
    u0 = torch.tensor([[1.0]])  # Initial displacement
    v0 = torch.tensor([[0.0]])  # Initial velocity

    loss_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Loss from physics-informed residual
        p_loss = physics_loss(model, t, omega)

        # Loss from initial conditions
        ic_loss = initial_condition_loss(model, t0, u0, v0)

        # Total loss
        loss = p_loss + ic_loss
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model, loss_history

# Define angular frequency
omega = 1.0  # For example, ω = 1 rad/s

# Train the PINN
model, loss_history = train_pinn(omega)

# Plotting the solution
t_test = torch.linspace(0, 10, 100).view(-1, 1)
u_pred = model(t_test).detach().numpy()

# Analytical solution for comparison
t_test_np = t_test.numpy()
u_exact = np.cos(omega * t_test_np)

plt.figure(figsize=(10, 5))
plt.plot(t_test_np, u_pred, label="PINN Solution")
plt.plot(t_test_np, u_exact, label="Exact Solution", linestyle="dashed")
plt.xlabel("Time (t)")
plt.ylabel("Displacement (u)")
plt.legend()
plt.title("PINN vs Analytical Solution for SHO")
plt.show()

# Plot loss history
plt.figure()
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss History")
plt.show()

