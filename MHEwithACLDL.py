import numpy as np
from numpy import random
from scipy.linalg import expm, solve, inv, cholesky
from scipy.signal import place_poles
import matplotlib.pyplot as plt
import time

def within_bounds(x, lb, ub, tol=1e-8):
    violations = (x < lb - tol) | (x > ub + tol)
    if np.any(violations):
        indices = np.argwhere(violations)
        for idx in indices:
            i, j = idx
            val = x[i, j]
            print(f"  Violation at t={i*Delta:.2f}s, x_{j} = {val:.6f}, bounds = [{lb[j]}, {ub[j]}]")
        return False
    return True

def log_violations(trial, xhat, lb, ub, tol=1e-8):
    for i in range(xhat.shape[0]):
        for j in range(xhat.shape[1]):
            val = xhat[i, j]
            if val < lb[j] - tol or val > ub[j] + tol:
                print(f"[Trial {trial}] Violation at t={i*Delta:.2f}s, x_{j} = {val:.6f}, bounds = [{lb[j]}, {ub[j]}]")

# --- System definition (continuous) ---
Acont = np.array([
    [-0.026, 0.074, -0.804, -9.809, 0],
    [-0.242, -2.017, 73.297, -0.105, -0.001],
    [0.003, -0.135, -2.941, 0, 0],
    [0, 0, 1, 0, 0],
    [-0.011, 1, 0, -75, 0]
])
Bcont = np.array([
    [4.594, 0],
    [-0.0004, -13.735],
    [0.0002, -24.410],
    [0, 0],
    [0, 0]
])
C = np.array([
    [1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
])

# Sampling
Delta = 0.1
Nx, Nu, Ny = 5, 2, 3
Nsim = 50
t = np.arange(Nsim+1) * Delta

# Discretize
A = expm(Acont*Delta)
# exact B discretization:

from scipy.linalg import solve_continuous_lyapunov
# simple zero‐order hold:
B = np.linalg.solve(Acont, (A - np.eye(Nx))).dot(Bcont)

# Noise covariances
Q = 0.01 * np.eye(Nx)
R = 0.1  * np.eye(Ny)

# Observer gain L (A-CL)
desired_poles = [0.6, 0.61, 0.62, 0.63, 0.64]
L = place_poles(A.T, C.T, desired_poles).gain_matrix.T

# Analytic MHE + pre-estimator
class LinearMHEPreEstimator:
    def __init__(self, A, B, C, Q, R, Pinv):
        self.A = A; self.B = B; self.C = C
        self.Qinv = inv(Q); self.Rinv = inv(R); self.Pinv = Pinv
        self.nx = A.shape[0]

    def estimate(self, x0bar, u_seq, y_seq):
        M = y_seq.shape[0] - 1
        n = self.nx
        Nvar = (M+1)*n
        H = np.zeros((Nvar, Nvar))
        g = np.zeros(Nvar)

        # Initial‐state prior
        H[0:n,0:n] += self.Pinv
        g[0:n]     += self.Pinv @ x0bar

        # Measurement terms
        for k in range(M+1):
            i = k*n
            H[i:i+n, i:i+n] += self.C.T @ self.Rinv @ self.C
            g[i:i+n]       += self.C.T @ self.Rinv @ y_seq[k]
        # Dynamics terms
        for k in range(M):
            i = k*n; j = (k+1)*n
            H[i:i+n, i:i+n] += self.A.T @ self.Qinv @ self.A
            H[i:i+n, j:j+n] += -self.A.T @ self.Qinv
            H[j:j+n, i:i+n] += -self.Qinv @ self.A
            H[j:j+n, j:j+n] += self.Qinv
            bu = self.B @ u_seq[k]
            g[i:i+n] += -self.A.T @ self.Qinv @ bu
            g[j:j+n] +=  self.Qinv @ bu
        from qpsolvers import solve_qp
        # Constraints [u, w, q, theta, h]
        state_lb = np.array([7.0, -10.0, -1.0, -0.5, 0.0])
        state_ub = np.array([100.0, 10.0, 1.0, 0.5, 200.0])

        lb = np.tile(state_lb, M+1)
        ub = np.tile(state_ub, M+1)

        lb = np.tile(state_lb, M+1)
        ub = np.tile(state_ub, M+1)

        G = np.vstack([
            np.eye(Nvar),      
            -np.eye(Nvar)     
        ])
        h = np.hstack([ub, -lb])


        from cvxopt import matrix as cvxmat

        # Solve QP: minimize 0.5 x^T H x - g^T x  s.t. G x <= h
        # DO NOT use cvxmat() with qpsolvers.solve_qp
        # Just reshape h properly and ensure float64 type
        x = solve_qp(
        H.astype(np.float64),
        -g.astype(np.float64),
        G.astype(np.float64),
        h.reshape(-1, 1).astype(np.float64),  # make sure h is (m,1)
        solver="mosek"
        )   



        if x is None:
            raise ValueError("QP solver failed to find a solution.")
        
        return x[-n:]  # return last state in window


# Precompute inverses
Qinv = inv(Q)
Rinv = inv(R)
P0 = np.eye(Nx)     
Pinv0 = inv(P0)

# Build estimator 
horizon = 25
mhe = LinearMHEPreEstimator(A, B, C, Q, R, Pinv0) 

random.seed(927)
x_true = np.zeros((Nsim+1, Nx))
y_meas = np.zeros((Nsim+1, Ny))
x_true[0] = np.zeros(Nx)
omega = 2*np.pi/(Nsim*Delta)
u = np.vstack([np.sin(omega*t), np.cos(omega*t)]).T[:-1]
w = cholesky(Q, lower=True) @ random.randn(Nx, Nsim)
v = cholesky(R, lower=True) @ random.randn(Ny, Nsim+1)
for k in range(Nsim+1):
    y_meas[k] = C @ x_true[k] + v[:,k]
    if k<Nsim:
        x_true[k+1] = A @ x_true[k] + B @ u[k] + w[:,k]

xhat = np.zeros((Nsim+1, Nx))
xhat[0] = np.array([20.0, 1.0, 0.1, 0.2, 100.0])  
x0bar = xhat[0].copy()
P = P0.copy()

times = []
total_start_time = time.perf_counter()

for k in range(Nsim):
    t0 = time.perf_counter()

    # Build window
    M = min(max(k,1), horizon)
    tmin = max(0, k-horizon)
    tmax = k+1
    u_win = u[tmin:tmax-1] if k>=1 else np.zeros((0, Nu))
    y_win = y_meas[tmin:tmax]

    # Pre‐estimator update (A-CL)
    if k>0:
        x0bar = A @ xhat[k] + B @ u[k-1] + L @ (y_meas[k] - C @ xhat[k])
        P = (A - L@C)@P@(A - L@C).T + Q + L@R@L.T
        mhe.Pinv = inv(P)

    # MHE analytic solve
    xhat[k+1] = mhe.estimate(x0bar, u_win, y_win)

    times.append(time.perf_counter() - t0)
total_end_time = time.perf_counter()

# --- Results ---
print(f"Avg. time per iter: {np.mean(times)*1e3:.6f} ms")
print(f"Total time for {Nsim} iterations: {(total_end_time - total_start_time)*1e3:.6f} ms")

# Plots
fig, axs = plt.subplots(Nx,1, figsize=(8,Nx*2))
for i in range(Nx):
    axs[i].plot(t, x_true[:,i], label=f"True $x_{i}$")
    axs[i].plot(t, xhat[:,i],'--', label=f"Est. $x_{i}$")
    axs[i].legend(); axs[i].set_ylabel(f"$x_{i}$")
axs[-1].set_xlabel("Time [s]")
plt.suptitle("Analytic Linear MHE + A-CL")
plt.tight_layout()
plt.show() 


# Monte Carlo dataset generation from MHE
num_simulations = 1000
horizon = 25
seq_length = 1  # Only 1-step MHE output is used

X_mhe_data = []
Y_mhe_data = []

for sim in range(num_simulations):
    x_sim = np.zeros((Nsim+1, Nx))
    x_sim[0] = np.random.uniform(-1, 1, size=Nx)
    u_sim = np.vstack([np.sin(omega*t), np.cos(omega*t)]).T[:-1]
    y_sim = np.zeros((Nsim+1, Ny))

    # Add noise
    w = cholesky(Q, lower=True) @ random.randn(Nx, Nsim)
    v = cholesky(R, lower=True) @ random.randn(Ny, Nsim+1)

    for k in range(Nsim):
        x_sim[k+1] = A @ x_sim[k] + B @ u_sim[k] + w[:, k]
        y_sim[k] = C @ x_sim[k] + v[:, k]
    y_sim[Nsim] = C @ x_sim[Nsim] + v[:, Nsim]

    xhat_sim = np.zeros((Nsim+1, Nx))
    x0bar = x_sim[0].copy()
    P = np.eye(Nx)
    
    for k in range(Nsim):
        tmin = max(0, k-horizon)
        tmax = k+1
        u_win = u_sim[tmin:tmax-1] if k >= 1 else np.zeros((0, Nu))
        y_win = y_sim[tmin:tmax]

        if k > 0:
            x0bar = A @ xhat_sim[k] + B @ u_sim[k-1] + L @ (y_sim[k] - C @ xhat_sim[k])
            P = (A - L @ C) @ P @ (A - L @ C).T + Q + L @ R @ L.T
            mhe.Pinv = inv(P)
        
        xhat_sim[k+1] = mhe.estimate(x0bar, u_win, y_win)

        # Save features and labels
        X_mhe_data.append(np.hstack([xhat_sim[k], u_sim[k], y_sim[k+1]]))  # input
        Y_mhe_data.append(xhat_sim[k+1])  # target

X_mhe_data = np.array(X_mhe_data)
Y_mhe_data = np.array(Y_mhe_data)

print("MHE dataset shapes:", X_mhe_data.shape, Y_mhe_data.shape)
# Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_mhe_data, Y_mhe_data, test_size=0.2, random_state=42)

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define NN
class MHE_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model = MHE_Net(X_train.shape[1], 64, Nx)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
def train_model(model, loader, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(loader):.4f}")

train_model(model, train_loader, optimizer, criterion)

# Run estimation using NN
xhat_nn = np.zeros((Nsim+1, Nx))
xhat_nn[0] = np.array([20.0, 1.0, 0.1, 0.2, 100.0])  

start_time_nn = time.perf_counter()

for k in range(Nsim):
    
    x_input = np.hstack([xhat_nn[k], u[k], y_meas[k+1]])
    x_input_scaled = scaler_X.transform(x_input.reshape(1, -1))
    x_input_tensor = torch.tensor(x_input_scaled, dtype=torch.float32)

    with torch.no_grad():
        x_next_scaled = model(x_input_tensor).numpy()
    x_next = scaler_y.inverse_transform(x_next_scaled)
    xhat_nn[k+1] = x_next   
end_time_nn = time.perf_counter()
avg_time_nn = (end_time_nn - start_time_nn) / Nsim * 1e3
print(f"Avg. NN estimation time per iter: {avg_time_nn:.6f} ms")
# Compare MHE vs NN estimation
plt.figure(figsize=(10, Nx*2))
for i in range(Nx):
    plt.subplot(Nx,1,i+1)
    plt.plot(t, x_true[:,i], label='True')
    plt.plot(t, xhat[:,i], '--', label='MHE')
    plt.plot(t, xhat_nn[:,i], ':', label='NN Est.')
    plt.ylabel(f'$x_{i}$')
    plt.legend()
plt.xlabel("Time [s]")
plt.suptitle("State Estimation: MHE vs NN Approximation")
plt.tight_layout()
plt.show()


def compute_rmse(xhat, xtrue):
    errors = np.sum((xhat - xtrue)**2, axis=1)
    rmse_total = np.sqrt(np.mean(errors))
    rmse_per_state = np.sqrt(np.mean((xhat - xtrue)**2, axis=0))
    return rmse_total, rmse_per_state

rmse_mhe_total, rmse_mhe_states = compute_rmse(xhat, x_true)
rmse_nn_total, rmse_nn_states = compute_rmse(xhat_nn, x_true)

print(f"\n--- RMSE Results ---")
print(f"MHE RMSE (total): {rmse_mhe_total:.6f}")
for i, v in enumerate(rmse_mhe_states):
    print(f"  x_{i} RMSE (MHE): {v:.6f}")
print(f"\nNN RMSE (total): {rmse_nn_total:.6f}")
for i, v in enumerate(rmse_nn_states):
    print(f"  x_{i} RMSE (NN): {v:.6f}")

# === MONTE CARLO EVALUATION ===
N_mc = 100
rng = np.random.default_rng(42)

results = {
    "MHE": {"rmse": [], "constraint_ok": [], "time": []},
    "NN":  {"rmse": [], "constraint_ok": [], "time": []}
}

print("\nRunning Monte Carlo simulations...")
state_lb = np.array([7.0, -10.0, -1.0, -0.5, 0.0])
state_ub = np.array([100.0, 10.0, 1.0, 0.5, 200.0])

for trial in range(N_mc):
    x0 = rng.uniform(low=[5, -1, -0.1, -0.1, 10], high=[20, 1, 0.1, 0.1, 50])
    x_true_mc = np.zeros((Nsim+1, Nx))
    y_meas_mc = np.zeros((Nsim+1, Ny))
    x_true_mc[0] = x0
    u_mc = np.vstack([np.sin(omega*t), np.cos(omega*t)]).T[:-1]

    w_mc = cholesky(Q, lower=True) @ rng.standard_normal((Nx, Nsim))
    v_mc = cholesky(R, lower=True) @ rng.standard_normal((Ny, Nsim+1))

    for k in range(Nsim):
        x_true_mc[k+1] = A @ x_true_mc[k] + B @ u_mc[k] + w_mc[:, k]
        y_meas_mc[k] = C @ x_true_mc[k] + v_mc[:, k]
    y_meas_mc[Nsim] = C @ x_true_mc[Nsim] + v_mc[:, Nsim]

    # --- MHE ---
    xhat_mhe = np.zeros((Nsim+1, Nx))
    xhat_mhe[0] = np.array([20.0, 1.0, 0.1, 0.2, 100.0])
    x0bar = xhat_mhe[0].copy()
    P = np.eye(Nx)
    times_mhe = []

    for k in range(Nsim):
        M = min(max(k, 1), horizon)
        tmin = max(0, k-horizon)
        u_win = u_mc[tmin:k] if k >= 1 else np.zeros((0, Nu))
        y_win = y_meas_mc[tmin:k+1]

        if k > 0:
            x0bar = A @ xhat_mhe[k] + B @ u_mc[k-1] + L @ (y_meas_mc[k] - C @ xhat_mhe[k])
            P = (A - L@C) @ P @ (A - L@C).T + Q + L@R@L.T
            mhe.Pinv = inv(P)

        t0 = time.perf_counter()
        xhat_mhe[k+1] = mhe.estimate(x0bar, u_win, y_win)
        times_mhe.append((time.perf_counter() - t0) * 1e3)

    rmse_mhe = compute_rmse(xhat_mhe, x_true_mc)[0]
    ok_mhe = within_bounds(xhat_mhe, state_lb, state_ub)
    if not ok_mhe:
        log_violations(trial, xhat_mhe, state_lb, state_ub)
    results["MHE"]["rmse"].append(rmse_mhe)
    results["MHE"]["constraint_ok"].append(ok_mhe)
    results["MHE"]["time"].append(times_mhe)

    # --- NN ---
    xhat_nn = np.zeros((Nsim+1, Nx))
    xhat_nn[0] = np.array([20.0, 1.0, 0.1, 0.2, 100.0])
    times_nn = []

    for k in range(Nsim):
        x_input = np.hstack([xhat_nn[k], u_mc[k], y_meas_mc[k+1]])
        x_input_scaled = scaler_X.transform(x_input.reshape(1, -1))
        x_input_tensor = torch.tensor(x_input_scaled, dtype=torch.float32)

        t0 = time.perf_counter()
        with torch.no_grad():
            x_next_scaled = model(x_input_tensor).numpy()
        times_nn.append((time.perf_counter() - t0) * 1e3)
        xhat_nn[k+1] = scaler_y.inverse_transform(x_next_scaled)

    rmse_nn = compute_rmse(xhat_nn, x_true_mc)[0]
    ok_nn = within_bounds(xhat_nn, state_lb, state_ub)
    results["NN"]["rmse"].append(rmse_nn)
    results["NN"]["constraint_ok"].append(ok_nn)
    results["NN"]["time"].append(times_nn)

    if (trial+1) % 10 == 0:
        print(f"  Completed {trial+1}/{N_mc} trials")

def summarize(est_name):
    rmse_vals = results[est_name]["rmse"]
    time_vals = np.array(results[est_name]["time"])
    ok_count = np.sum(results[est_name]["constraint_ok"])
    print(f"\n--- {est_name} Summary ---")
    print(f"Avg. RMSE: {np.mean(rmse_vals):.6f}")
    print(f"Constraints satisfied: {ok_count}/{N_mc}")
    print(f"Computation time per step [ms]: min={np.min(time_vals):.3f}, max={np.max(time_vals):.3f}, mean={np.mean(time_vals):.3f}")

summarize("MHE")
summarize("NN")

