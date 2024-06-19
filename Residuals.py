import numpy as np
import matplotlib.pyplot as plt

# Function to compute residuals
def compute_residuals(T, dx):
    residuals = np.zeros_like(T)
    for i in range(1, len(T)-1):
        residuals[i] = T[i+1] - 2*T[i] + T[i-1]
    return residuals / dx**2

# Function to perform Gauss-Seidel iteration
def gauss_seidel(T, dx, max_iter, tolerance, omega=1.0):
    residuals_history = []
    temperature_history = [T.copy()]
    for iteration in range(max_iter):
        T_old = T.copy()
        for i in range(1, len(T)-1):
            T[i] = (1-omega) * T[i] + omega * 0.5 * (T[i-1] + T[i+1])
        
        # Compute residuals
        residuals = compute_residuals(T, dx)
        max_residual = np.max(np.abs(residuals))
        residuals_history.append(max_residual)
        temperature_history.append(T.copy())
        
        print(f"Iteration {iteration+1}, Max Residual: {max_residual}")
        
        # Check convergence
        if max_residual < tolerance:
            print(f"Converged in {iteration+1} iterations")
            break
    return T, residuals_history, temperature_history

# Plotting function with enhanced font sizes
def plot_results(x, T, residuals_history, temperature_history, title):
    plt.figure(figsize=(14, 14))
    
    font_size = 22
    tick_font_size = 22

    plt.subplot(3, 1, 1)
    plt.plot(x, T, 'bo-', label='Temperature')
    plt.xlabel('x', fontsize=font_size)
    plt.ylabel('Temperature', fontsize=font_size)
    plt.title(f'1D Steady-State Heat Conduction - {title}', fontsize=font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.plot(residuals_history, 'ro-', label='Max Residual')
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('Residual', fontsize=font_size)
    plt.yscale('log')
    plt.title('Residuals History', fontsize=font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.grid(True)
    plt.tight_layout()



    plt.tight_layout()
    plt.show()

# Case 1: Poor Initial Guess
L = 1.0
N = 10
dx = L / (N-1)
T_left = 100.0
T_right = 0.0
T_poor = np.random.rand(N) * 100
T_poor[0] = T_left
T_poor[-1] = T_right
T_poor, residuals_history_poor, temperature_history_poor = gauss_seidel(T_poor, dx, max_iter=1000, tolerance=1e-6)
plot_results(np.linspace(0, L, N), T_poor, residuals_history_poor, temperature_history_poor, 'Poor Initial Guess')

# Case 2: Higher Grid Resolution
N_high = 50
dx_high = L / (N_high-1)
T_high = np.ones(N_high) * (T_left + T_right) / 2
T_high[0] = T_left
T_high[-1] = T_right
T_high, residuals_history_high, temperature_history_high = gauss_seidel(T_high, dx_high, max_iter=1000, tolerance=1e-6)
plot_results(np.linspace(0, L, N_high), T_high, residuals_history_high, temperature_history_high, 'Higher Grid Resolution')

# Case 3: Over-relaxation
T_relax = np.ones(N) * (T_left + T_right) / 2
T_relax[0] = T_left
T_relax[-1] = T_right
omega = 1.5  # Over-relaxation factor
T_relax, residuals_history_relax, temperature_history_relax = gauss_seidel(T_relax, dx, max_iter=1000, tolerance=1e-6, omega=omega)
plot_results(np.linspace(0, L, N), T_relax, residuals_history_relax, temperature_history_relax, 'Over-relaxation (ω=1.5)')
