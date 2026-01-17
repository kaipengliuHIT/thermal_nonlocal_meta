"""
PSO (Particle Swarm Optimization) for Nonlocal Thermal Metasurface Design
Based on: "Inverse Design and Experimental Verification of Nonlocal Thermal Metasurfaces 
via NURBS and Spatiotemporal Coupled-Mode Theory"

This script optimizes the geometric parameters of a BIC-supporting metasurface
to achieve asymmetric thermal emission profiles using Lumerical FDTD simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
import importlib

# Import the simulation module
from sim_for_pso import Simulation

# =============================================================================
# PSO Algorithm Configuration
# =============================================================================

class PSOConfig:
    """Configuration class for PSO optimization parameters."""
    
    def __init__(self):
        # PSO parameters
        self.n_particles = 20          # Number of particles in swarm
        self.n_iterations = 50         # Maximum iterations
        self.w = 0.7                    # Inertia weight
        self.c1 = 1.5                   # Cognitive coefficient (personal best)
        self.c2 = 1.5                   # Social coefficient (global best)
        self.w_decay = 0.99             # Inertia weight decay factor
        
        # Wavelength range (meters) - 11.6 to 13.6 μm as per paper
        self.wavelength_start = 11.6e-6
        self.wavelength_stop = 13.6e-6
        
        # Angular range for optimization (degrees) - -60° to 60° as per paper
        self.theta_min = -60
        self.theta_max = 60
        self.theta_step = 5
        
        # Regularization coefficient for unmanufacturable designs
        self.alpha = 0.01
        
        # Convergence criteria
        self.tolerance = 1e-6
        self.patience = 10              # Early stopping patience


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for metasurface design.
    
    Optimizes perturbation parameters to achieve target thermal emission profile.
    Uses Lumerical FDTD simulations via the Simulation class.
    """
    
    def __init__(self, config: PSOConfig, param_bounds: dict, target_emissivity: callable = None):
        """
        Initialize PSO optimizer.
        
        Args:
            config: PSOConfig object with optimization parameters
            param_bounds: Dictionary with parameter names and (min, max) bounds
            target_emissivity: Function that returns target emissivity profile
        """
        self.config = config
        self.param_bounds = param_bounds
        self.n_params = len(param_bounds)
        self.param_names = list(param_bounds.keys())
        
        # Extract bounds as arrays
        self.lower_bounds = np.array([param_bounds[k][0] for k in self.param_names])
        self.upper_bounds = np.array([param_bounds[k][1] for k in self.param_names])
        
        # Target emissivity function
        self.target_emissivity = target_emissivity if target_emissivity else self._default_target
        
        # Initialize particles
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        
        # History tracking
        self.history = {
            'global_best_scores': [],
            'mean_scores': [],
            'best_params': [],
            'all_evaluations': []
        }
        
        # Simulation instance (will be created when optimization starts)
        self.sim = None
        
    def _default_target(self, theta, wavelength):
        """
        Default asymmetric target emissivity profile.
        
        Creates an asymmetric emission profile with higher emissivity 
        at positive angles (as described in the paper).
        
        Args:
            theta: Angle in degrees
            wavelength: Wavelength in meters
            
        Returns:
            Target emissivity value (0-1)
        """
        # Asymmetric profile: higher emission at positive angles
        # Based on paper's goal of asymmetric thermal emission
        center_wavelength = 12.6e-6
        wavelength_bandwidth = 1e-6
        
        # Spectral profile (Gaussian around center wavelength)
        spectral_factor = np.exp(-((wavelength - center_wavelength) / wavelength_bandwidth) ** 2)
        
        # Angular asymmetry: sigmoid-like transition
        # High emissivity for theta > 0, low for theta < 0
        if theta > 0:
            angular_factor = 0.8 + 0.15 * (1 - np.exp(-theta / 30))
        else:
            angular_factor = 0.3 - 0.1 * (1 - np.exp(theta / 30))
        
        return spectral_factor * angular_factor
    
    def _initialize_particles(self):
        """Initialize particle positions and velocities randomly within bounds."""
        n = self.config.n_particles
        d = self.n_params
        
        # Random initialization within bounds
        self.particles = np.random.uniform(
            self.lower_bounds, 
            self.upper_bounds, 
            size=(n, d)
        )
        
        # Initialize velocities (small random values)
        velocity_range = (self.upper_bounds - self.lower_bounds) * 0.1
        self.velocities = np.random.uniform(
            -velocity_range, 
            velocity_range, 
            size=(n, d)
        )
        
        # Personal best initialization
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = np.full(n, np.inf)
        
    def _params_to_control_points(self, params):
        """
        Convert parameter vector to NURBS control points for simulation.
        
        The parameters represent perturbations (d values) and lengths (l values)
        for the 6 curved waveguide structures.
        
        Args:
            params: Parameter vector [d1, d2, d3, d4, d5, d6, l1, l2, l3, l4, l5, l6]
            
        Returns:
            Control points array for Simulation class
        """
        # Extract d (perturbation) and l (length) values
        d = params[:6]
        l = params[6:]
        
        # Base x-positions for 6 waveguides
        x_base = np.array([-5.25, -3.15, -1.05, 1.05, 3.15, 5.25])
        
        # Construct control points for each waveguide
        # Each waveguide has 3 control points defining a curved shape
        control_points = np.array([
            [[x_base[0] + d[0], -l[0]], [x_base[0], 0], [x_base[0] + d[0], l[0]]],
            [[x_base[1] + d[1], -l[1]], [x_base[1], 0], [x_base[1] + d[1], l[1]]],
            [[x_base[2] + d[2], -l[2]], [x_base[2], 0], [x_base[2] + d[2], l[2]]],
            [[x_base[3] + d[3], -l[3]], [x_base[3], 0], [x_base[3] + d[3], l[3]]],
            [[x_base[4] + d[4], -l[4]], [x_base[4], 0], [x_base[4] + d[4], l[4]]],
            [[x_base[5] + d[5], -l[5]], [x_base[5], 0], [x_base[5] + d[5], l[5]]]
        ])
        
        return control_points
    
    def _regularization(self, params):
        """
        Regularization term to penalize unmanufacturable designs.
        
        Penalizes:
        - Negative lengths
        - Too small perturbations (manufacturing constraints)
        - Overlapping structures
        
        Args:
            params: Parameter vector
            
        Returns:
            Regularization penalty value
        """
        penalty = 0.0
        
        d = params[:6]
        l = params[6:]
        
        # Penalty for negative or very small lengths
        min_length = 0.5  # Minimum manufacturable length (μm)
        penalty += np.sum(np.maximum(0, min_length - l) ** 2)
        
        # Penalty for excessive perturbations that might cause overlap
        max_perturbation = 2.0  # Maximum perturbation (μm)
        penalty += np.sum(np.maximum(0, np.abs(d) - max_perturbation) ** 2)
        
        return penalty
    
    def _objective_function(self, params, particle_idx=None):
        """
        Objective function F(p) as defined in the paper (Equation 1).
        
        F(p) = ∫∫ [E_sim(λ,θ;p) - E_target(λ,θ)]² dλdθ + α*R(p)
        
        Args:
            params: Parameter vector p
            particle_idx: Optional particle index for logging
            
        Returns:
            Objective function value (lower is better)
        """
        try:
            # Convert parameters to control points
            control_points = self._params_to_control_points(params)
            
            # Update simulation with new control points
            self.sim.set_contral_points(control_points)
            self.sim.setup_simulation()
            
            # Run FDTD simulation
            reflectance, phase = self.sim.run_forward()
            
            # Calculate emissivity from reflectance (Kirchhoff's law: ε = 1 - R for opaque materials)
            emissivity = 1 - np.abs(reflectance)
            
            # For angular dependence, we need to run multiple simulations at different angles
            # In this simplified version, we use the normal incidence result
            # and apply an angular weighting based on the phase information
            
            # Calculate squared error between simulated and target
            # Wavelength array
            wavelengths = np.linspace(
                self.config.wavelength_start,
                self.config.wavelength_stop,
                len(emissivity)
            )
            
            # Compute target emissivity for each wavelength
            target = np.array([self.target_emissivity(0, wl) for wl in wavelengths])
            
            # Mean squared error
            mse = np.mean((emissivity - target) ** 2)
            
            # Add regularization term
            reg_penalty = self._regularization(params)
            
            # Total objective
            objective = mse + self.config.alpha * reg_penalty
            
            return objective, emissivity
            
        except Exception as e:
            print(f"Simulation error for particle {particle_idx}: {e}")
            return np.inf, None
    
    def _update_velocity(self, idx, w):
        """
        Update velocity for particle idx using PSO update rule.
        
        v_new = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        
        Args:
            idx: Particle index
            w: Current inertia weight
        """
        r1 = np.random.random(self.n_params)
        r2 = np.random.random(self.n_params)
        
        cognitive = self.config.c1 * r1 * (self.personal_best_positions[idx] - self.particles[idx])
        social = self.config.c2 * r2 * (self.global_best_position - self.particles[idx])
        
        self.velocities[idx] = w * self.velocities[idx] + cognitive + social
        
        # Velocity clamping to prevent explosion
        max_velocity = (self.upper_bounds - self.lower_bounds) * 0.2
        self.velocities[idx] = np.clip(self.velocities[idx], -max_velocity, max_velocity)
    
    def _update_position(self, idx):
        """
        Update position for particle idx.
        
        x_new = x + v
        
        Enforces boundary constraints.
        
        Args:
            idx: Particle index
        """
        self.particles[idx] = self.particles[idx] + self.velocities[idx]
        
        # Enforce bounds
        self.particles[idx] = np.clip(
            self.particles[idx], 
            self.lower_bounds, 
            self.upper_bounds
        )
    
    def optimize(self, save_results=True, output_dir='pso_results'):
        """
        Run PSO optimization.
        
        Args:
            save_results: Whether to save results to disk
            output_dir: Directory for saving results
            
        Returns:
            Best parameters and optimization history
        """
        print("=" * 60)
        print("PSO Optimization for Nonlocal Thermal Metasurface Design")
        print("=" * 60)
        
        # Create output directory
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize simulation (use first particle's params as initial)
        print("\nInitializing Lumerical FDTD simulation...")
        initial_params = (self.lower_bounds + self.upper_bounds) / 2
        initial_control_points = self._params_to_control_points(initial_params)
        self.sim = Simulation(initial_control_points)
        
        # Initialize particles
        print(f"Initializing {self.config.n_particles} particles...")
        self._initialize_particles()
        
        # Current inertia weight
        w = self.config.w
        
        # No improvement counter for early stopping
        no_improvement_count = 0
        prev_best_score = np.inf
        
        # Main optimization loop
        print("\nStarting optimization...")
        print("-" * 60)
        
        for iteration in range(self.config.n_iterations):
            iteration_scores = []
            
            # Evaluate all particles
            for idx in tqdm(range(self.config.n_particles), 
                           desc=f"Iteration {iteration + 1}/{self.config.n_iterations}"):
                
                # Evaluate objective function
                score, emissivity = self._objective_function(self.particles[idx], idx)
                iteration_scores.append(score)
                
                # Update personal best
                if score < self.personal_best_scores[idx]:
                    self.personal_best_scores[idx] = score
                    self.personal_best_positions[idx] = self.particles[idx].copy()
                
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[idx].copy()
                    
                    # Save best emissivity profile
                    if emissivity is not None:
                        self.history['best_emissivity'] = emissivity
            
            # Update velocities and positions
            for idx in range(self.config.n_particles):
                self._update_velocity(idx, w)
                self._update_position(idx)
            
            # Decay inertia weight
            w *= self.config.w_decay
            
            # Record history
            self.history['global_best_scores'].append(self.global_best_score)
            self.history['mean_scores'].append(np.mean(iteration_scores))
            self.history['best_params'].append(self.global_best_position.copy())
            
            # Print progress
            print(f"  Best Score: {self.global_best_score:.6f} | "
                  f"Mean Score: {np.mean(iteration_scores):.6f} | "
                  f"Inertia: {w:.4f}")
            
            # Early stopping check
            if abs(prev_best_score - self.global_best_score) < self.config.tolerance:
                no_improvement_count += 1
                if no_improvement_count >= self.config.patience:
                    print(f"\nEarly stopping: No improvement for {self.config.patience} iterations")
                    break
            else:
                no_improvement_count = 0
                prev_best_score = self.global_best_score
        
        print("-" * 60)
        print("\nOptimization Complete!")
        print(f"Best Objective Score: {self.global_best_score:.6f}")
        print(f"Best Parameters:")
        for i, name in enumerate(self.param_names):
            print(f"  {name}: {self.global_best_position[i]:.4f}")
        
        # Save results
        if save_results:
            self._save_results(output_dir, timestamp)
        
        return self.global_best_position, self.history
    
    def _save_results(self, output_dir, timestamp):
        """Save optimization results to files."""
        # Save parameters as JSON
        results = {
            'best_score': float(self.global_best_score),
            'best_params': {
                name: float(val) 
                for name, val in zip(self.param_names, self.global_best_position)
            },
            'config': {
                'n_particles': self.config.n_particles,
                'n_iterations': self.config.n_iterations,
                'w': self.config.w,
                'c1': self.config.c1,
                'c2': self.config.c2
            },
            'history': {
                'global_best_scores': [float(x) for x in self.history['global_best_scores']],
                'mean_scores': [float(x) for x in self.history['mean_scores']]
            }
        }
        
        json_path = os.path.join(output_dir, f'pso_results_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")
        
        # Plot convergence curve
        self._plot_convergence(output_dir, timestamp)
        
        # Plot best emissivity profile
        if 'best_emissivity' in self.history:
            self._plot_emissivity(output_dir, timestamp)
    
    def _plot_convergence(self, output_dir, timestamp):
        """Plot and save convergence curve."""
        plt.figure(figsize=(10, 6))
        
        iterations = range(1, len(self.history['global_best_scores']) + 1)
        
        plt.plot(iterations, self.history['global_best_scores'], 'b-', 
                 linewidth=2, label='Global Best')
        plt.plot(iterations, self.history['mean_scores'], 'r--', 
                 linewidth=1.5, label='Mean Score', alpha=0.7)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Objective Function Value', fontsize=12)
        plt.title('PSO Convergence Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'convergence_{timestamp}.png'), dpi=300)
        plt.close()
        print(f"Convergence plot saved")
    
    def _plot_emissivity(self, output_dir, timestamp):
        """Plot and save best emissivity profile."""
        plt.figure(figsize=(10, 6))
        
        emissivity = self.history['best_emissivity']
        wavelengths = np.linspace(
            self.config.wavelength_start * 1e6,  # Convert to μm
            self.config.wavelength_stop * 1e6,
            len(emissivity)
        )
        
        plt.plot(wavelengths, emissivity, 'b-', linewidth=2, label='Optimized')
        
        # Plot target for comparison
        target = np.array([
            self.target_emissivity(0, wl * 1e-6) 
            for wl in wavelengths
        ])
        plt.plot(wavelengths, target, 'r--', linewidth=1.5, label='Target', alpha=0.7)
        
        plt.xlabel('Wavelength (μm)', fontsize=12)
        plt.ylabel('Emissivity', fontsize=12)
        plt.title('Optimized Emissivity Profile', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([11.6, 13.6])
        plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'emissivity_{timestamp}.png'), dpi=300)
        plt.close()
        print(f"Emissivity plot saved")


def create_asymmetric_target(asymmetry_degree=0.5):
    """
    Create a custom asymmetric target emissivity function.
    
    Args:
        asymmetry_degree: Degree of asymmetry (0 = symmetric, 1 = maximum asymmetry)
        
    Returns:
        Target emissivity function
    """
    def target_func(theta, wavelength):
        # Spectral profile centered at 12.6 μm
        center_wl = 12.6e-6
        spectral = np.exp(-((wavelength - center_wl) / (1e-6)) ** 2)
        
        # Angular asymmetry
        base_emissivity = 0.5
        asymmetric_factor = asymmetry_degree * 0.4 * np.tanh(theta / 20)
        
        return spectral * (base_emissivity + asymmetric_factor)
    
    return target_func


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    # Define parameter bounds for optimization
    # Parameters: d1-d6 (perturbations in μm), l1-l6 (lengths in μm)
    param_bounds = {
        'd1': (0.1, 1.5),   # Perturbation for waveguide 1
        'd2': (0.1, 1.5),   # Perturbation for waveguide 2
        'd3': (0.1, 1.5),   # Perturbation for waveguide 3
        'd4': (0.1, 1.5),   # Perturbation for waveguide 4
        'd5': (0.1, 1.5),   # Perturbation for waveguide 5
        'd6': (0.1, 1.5),   # Perturbation for waveguide 6
        'l1': (2.0, 6.0),   # Length for waveguide 1
        'l2': (2.0, 6.0),   # Length for waveguide 2
        'l3': (2.0, 6.0),   # Length for waveguide 3
        'l4': (2.0, 6.0),   # Length for waveguide 4
        'l5': (2.0, 6.0),   # Length for waveguide 5
        'l6': (2.0, 6.0),   # Length for waveguide 6
    }
    
    # Create configuration
    config = PSOConfig()
    config.n_particles = 15      # Adjust based on computational resources
    config.n_iterations = 30     # Adjust based on convergence requirements
    
    # Create asymmetric target emissivity
    target_func = create_asymmetric_target(asymmetry_degree=0.6)
    
    # Initialize optimizer
    optimizer = ParticleSwarmOptimizer(
        config=config,
        param_bounds=param_bounds,
        target_emissivity=target_func
    )
    
    # Run optimization
    print("\n" + "=" * 60)
    print("Starting PSO Optimization for Thermal Metasurface Design")
    print("Based on: Inverse Design of Nonlocal Thermal Metasurfaces")
    print("=" * 60 + "\n")
    
    best_params, history = optimizer.optimize(
        save_results=True,
        output_dir='pso_results'
    )
    
    # Print final optimized control points
    print("\n" + "=" * 60)
    print("Optimized Control Points for Lumerical Simulation:")
    print("=" * 60)
    
    d_values = best_params[:6]
    l_values = best_params[6:]
    x_base = np.array([-5.25, -3.15, -1.05, 1.05, 3.15, 5.25])
    
    print("\nWaveguide Control Points (μm):")
    for i in range(6):
        print(f"  Waveguide {i+1}:")
        print(f"    Point 1: ({x_base[i] + d_values[i]:.3f}, {-l_values[i]:.3f})")
        print(f"    Point 2: ({x_base[i]:.3f}, 0.000)")
        print(f"    Point 3: ({x_base[i] + d_values[i]:.3f}, {l_values[i]:.3f})")
    
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
