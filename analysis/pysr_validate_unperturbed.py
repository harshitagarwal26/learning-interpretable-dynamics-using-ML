import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor

# Add parent dir to path so we can import from examples and lag_caVAE
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.append(PARENT_DIR)

from examples.pend_lag_cavae_trainer import Model


def main():
    # 1. Load the unperturbed model
    ckpt_path = os.path.join(PARENT_DIR, 'results', 'pend', 'pend-lag-cavae-T_p=4-epoch=983-step=7871.ckpt')
    print(f"Loading unperturbed model from {ckpt_path}...")
    
    # Load model on CPU
    model = Model.load_from_checkpoint(ckpt_path, map_location='cpu')
    model.eval()
    
    # 2. Sample theta values
    n_samples = 500
    theta = np.linspace(-np.pi, np.pi, n_samples)
    
    # Compute cos and sin for V_net
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    q_input = torch.tensor(np.stack([cos_theta, sin_theta], axis=1), dtype=torch.float32)
    
    # Evaluate V_net
    with torch.no_grad():
        V_pred = model.ode.V_net(q_input).numpy().flatten()
    
    # Also evaluate M_net to see the scale
    with torch.no_grad():
        M_pred = model.ode.M_net(q_input).numpy().flatten()
    
    print(f"Average learned M: {np.mean(M_pred):.4f}")
    
    # True theoretical potential derived from dynamics: V(θ) = -15 * M * cos(θ)
    V_true = -15 * np.mean(M_pred) * cos_theta
    print(f"True Dynamics Potential Amplitude: {-15 * np.mean(M_pred):.4f}")
    
    # Create output directory
    out_dir = os.path.join(THIS_DIR, 'outputs', 'pysr_unperturbed')
    os.makedirs(out_dir, exist_ok=True)
    
    # 3. PySR setup
    print("\nRunning PySR on θ -> V(θ)...")
    
    X = theta.reshape(-1, 1)  # input: angle
    y = V_pred                # target: learned V
    
    pysr_model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*"],
        unary_operators=["cos", "sin"],
        denoise=False,
        model_selection="best",
        maxsize=15,
        parsimony=0.01,
        random_state=42,
        procs=4,
        temp_equation_file=True,
    )
    
    pysr_model.fit(X, y, variable_names=["theta"])
    
    # --- ELBOW METHOD FOR EQUATION SELECTION ---
    # Instead of trusting PySR's "score" which can overfit on tiny drops in loss,
    # we mathematically find the "elbow" or "knee" of the log-loss vs complexity curve.
    equations = pysr_model.equations_
    
    complexities = equations['complexity'].values.astype(float)
    # Use log loss because the drop is typically exponential initially
    log_losses = np.log(equations['loss'].values.astype(float) + 1e-10)
    
    # Define the secant line connecting the simplest equation to the most complex
    p1 = np.array([complexities[0], log_losses[0]])
    p2 = np.array([complexities[-1], log_losses[-1]])
    
    distances = []
    for i in range(len(complexities)):
        p0 = np.array([complexities[i], log_losses[i]])
        # Calculate the orthogonal distance from point p0 to the secant line (p1 -> p2)
        d = np.abs(np.cross(p2 - p1, p1 - p0)) / np.linalg.norm(p2 - p1)
        distances.append(d)
        
    # The elbow is the point furthest from the secant line
    elbow_idx = np.argmax(distances)
    best_idx = equations.index[elbow_idx]
    
    print(f"\nElbow method selected equation complexity: {complexities[elbow_idx]}")
    
    # Get the best equation string using the selected index
    best_equation_expr = pysr_model.sympy(index=best_idx)
    # Replace x0 with theta in the string representation if it shows up
    best_equation = str(best_equation_expr).replace('x0', 'theta')
    
    # Predict using that specific, structurally simple equation
    pysr_pred = pysr_model.predict(X, index=best_idx)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(theta, V_pred, 'b-', linewidth=2, label='V_net (Neural Network)')
    plt.plot(theta, pysr_pred, 'g--', linewidth=2, label=f'PySR (Complexity {equations.loc[best_idx, "complexity"]}): {best_equation}')
    plt.plot(theta, V_true, 'r:', linewidth=2, label='True Dynamics Potential (-15*M*cosθ)')
    plt.xlabel('Angle θ (rad)')
    plt.ylabel('Potential Energy V(θ)')
    plt.title('Comparison of Potentials')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=10)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'comparison.png'))
    print(f"Saved plot to {out_dir}/comparison.png")
    
    with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
        f.write("PySR Validation on Unperturbed V_net\n")
        f.write("======================================\n\n")
        f.write(f"Average learned M: {np.mean(M_pred):.4f}\n")
        f.write(f"True Dynamics Potential Amplitude: {-15 * np.mean(M_pred):.4f}\n\n")
        
        f.write("PySR Results:\n")
        f.write(pysr_model.equations_.to_string())
        f.write(f"\n\nBest Equation Structure (via Elbow Method, Complexity {complexities[elbow_idx]}):\n")
        f.write(best_equation)
        f.write("\n")
        
    print(f"Saved report to {out_dir}/report.txt")


if __name__ == "__main__":
    main()
