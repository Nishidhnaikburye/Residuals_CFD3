import numpy as np
import matplotlib.pyplot as plt

# Function to compute residuals
def compute_residuals(T, dx):
    residuals = np.zeros_like(T)
    for i in range(1, len(T)-1):
        residuals[i] = T[i+1] - 2*T[i] + T[i-1]
    return residuals / dx**2

# Function to perform Gauss-Seidel iteration and calculate norms
def gauss_seidel(T, dx, max_iter, tolerance, omega=1.0):
    residuals_history = []
    l1_norms = []
    l2_norms = []
    linf_norms = []
    for iteration in range(max_iter):
        T_old = T.copy()
        for i in range(1, len(T)-1):
            T[i] = (1-omega) * T[i] + omega * 0.5 * (T[i-1] + T[i+1])
        
        # Compute residuals
        residuals = compute_residuals(T, dx)
        max_residual = np.max(np.abs(residuals))
        residuals_history.append(max_residual)
        
        # Calculate norms of residuals
        l1_norm = np.sum(np.abs(residuals))
        l2_norm = np.linalg.norm(residuals)
        linf_norm = np.max(np.abs(residuals))
        
        # Store norms
        l1_norms.append(l1_norm)
        l2_norms.append(l2_norm)
        linf_norms.append(linf_norm)
        
        print(f"Iteration {iteration+1}, Max Residual: {max_residual}, L1 Norm: {l1_norm}, L2 Norm: {l2_norm}, Linf Norm: {linf_norm}")
        
        # Check convergence
        if max_residual < tolerance:
            print(f"Converged in {iteration+1} iterations")
            break
    return T, residuals_history, l1_norms, l2_norms, linf_norms

# Plotting function
def plot_results(x, T, residuals_history, l1_norms, l2_norms, linf_norms, title):
    plt.figure(figsize=(14, 10))
    
    font_size = 22
    tick_font_size = 22

    plt.subplot(4, 1, 1)
    plt.plot(x, T, 'o-', label='Temperature')
    plt.xlabel('x', fontsize=font_size)
    plt.ylabel('Temperature', fontsize=font_size)
    plt.title(f'1D Steady-State Heat Conduction - {title}', fontsize=font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(4, 1, 2)
    plt.plot(residuals_history, 'o-', label='Max Residual')
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('Residual', fontsize=font_size)
    plt.yscale('log')
    plt.title('Residuals History', fontsize=font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(4, 1, 3)
    plt.plot(l1_norms, 'o-', label='L1 Norm')
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('L1 Norm', fontsize=font_size)
    plt.title('L1 Norm Evolution', fontsize=font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(4, 1, 4)
    plt.plot(l2_norms, 'o-', label='L2 Norm')
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('L2 Norm', fontsize=font_size)
    plt.title('L2 Norm Evolution', fontsize=font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('normsPlot1.png', dpi=500)


    plt.figure(figsize=(14, 10))
    plt.plot(linf_norms, 'o-', label='L∞ Norm')
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('L∞ Norm', fontsize=font_size)
    plt.title('L∞ Norm Evolution', fontsize=font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('normsPlot2.png', dpi=500)
    plt.show()

# Example usage:
if __name__ == "__main__":
    L = 1.0
    N = 50
    dx = L / (N-1)
    T_left = 100.0
    T_right = 0.0
    T_initial = np.ones(N) * (T_left + T_right) / 2
    T_initial[0] = T_left
    T_initial[-1] = T_right
    
    T_final, residuals_history, l1_norms, l2_norms, linf_norms = gauss_seidel(T_initial, dx, max_iter=1000, tolerance=1e-6)
    plot_results(np.linspace(0, L, N), T_final, residuals_history, l1_norms, l2_norms, linf_norms, 'Gauss-Seidel Method')
