"""
Training Data Generation Pipeline for STCMT Surrogate Model

This script generates training data by:
1. Randomly sampling NURBS control point configurations
2. Running single-period FDTD simulations
3. Extracting STCMT parameters from simulation results
4. Saving dataset for surrogate model training

Based on: "Inverse Design and Experimental Verification of Nonlocal Thermal Metasurfaces
via NURBS and Spatiotemporal Coupled-Mode Theory"
"""

import numpy as np
import json
import os
from datetime import datetime
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial

# Import simulation and extraction modules
from sim_stcmt_extraction import (
    STCMTSimulation,
    generate_random_nurbs_control_points,
    fit_resonance,
    extract_band_dispersion
)


# =============================================================================
# NURBS Control Point Sampling Strategies
# =============================================================================

class NURBSSampler:
    """
    Sampling strategies for NURBS control points.
    
    Provides various methods to generate diverse control point configurations
    for comprehensive training data coverage.
    """
    
    def __init__(self, 
                 x_range=(-6, 6),
                 y_range=(-5, 5),
                 n_curves=1,
                 n_points_range=(4, 8)):
        """
        Args:
            x_range: (min, max) range for x coordinates (μm)
            y_range: (min, max) range for y coordinates (μm)
            n_curves: Number of curves per meta-atom
            n_points_range: (min, max) control points per curve
        """
        self.x_range = x_range
        self.y_range = y_range
        self.n_curves = n_curves
        self.n_points_range = n_points_range
    
    def sample_random(self):
        """
        Generate completely random control points.
        
        Returns:
            List of control point arrays
        """
        curves = []
        for _ in range(self.n_curves):
            n_points = np.random.randint(
                self.n_points_range[0], 
                self.n_points_range[1] + 1
            )
            
            # Generate sorted x values for a smooth curve
            x_vals = np.sort(np.random.uniform(
                self.x_range[0], 
                self.x_range[1], 
                n_points
            ))
            
            y_vals = np.random.uniform(
                self.y_range[0], 
                self.y_range[1], 
                n_points
            )
            
            points = [[float(x), float(y)] for x, y in zip(x_vals, y_vals)]
            curves.append(points)
        
        return curves
    
    def sample_smooth_curve(self, smoothness=0.5):
        """
        Generate smooth curves with controlled curvature.
        
        Args:
            smoothness: 0-1, higher means smoother curves
            
        Returns:
            List of control point arrays
        """
        curves = []
        for _ in range(self.n_curves):
            n_points = np.random.randint(
                self.n_points_range[0], 
                self.n_points_range[1] + 1
            )
            
            # Generate base curve
            t = np.linspace(0, 1, n_points)
            x_vals = self.x_range[0] + t * (self.x_range[1] - self.x_range[0])
            
            # Generate smooth y values using sinusoidal basis
            y_base = np.zeros(n_points)
            n_harmonics = max(1, int((1 - smoothness) * 3) + 1)
            
            for h in range(n_harmonics):
                amplitude = np.random.uniform(-1, 1) * (self.y_range[1] - self.y_range[0]) / (h + 1)
                phase = np.random.uniform(0, 2 * np.pi)
                y_base += amplitude * np.sin(2 * np.pi * (h + 1) * t + phase)
            
            # Scale to y_range
            y_vals = np.clip(y_base, self.y_range[0], self.y_range[1])
            
            points = [[float(x), float(y)] for x, y in zip(x_vals, y_vals)]
            curves.append(points)
        
        return curves
    
    def sample_symmetric(self, symmetry_type='mirror'):
        """
        Generate symmetric control point configurations.
        
        Args:
            symmetry_type: 'mirror' (y-axis) or 'point' (origin)
            
        Returns:
            List of control point arrays
        """
        curves = []
        for _ in range(self.n_curves):
            # Generate half of the points
            n_half = np.random.randint(2, 5)
            
            x_half = np.sort(np.random.uniform(0, self.x_range[1], n_half))
            y_half = np.random.uniform(self.y_range[0], self.y_range[1], n_half)
            
            if symmetry_type == 'mirror':
                # Mirror across y-axis
                x_full = np.concatenate([-x_half[::-1], x_half])
                y_full = np.concatenate([y_half[::-1], y_half])
            else:  # point symmetry
                x_full = np.concatenate([-x_half[::-1], x_half])
                y_full = np.concatenate([-y_half[::-1], y_half])
            
            points = [[float(x), float(y)] for x, y in zip(x_full, y_full)]
            curves.append(points)
        
        return curves
    
    def sample_perturbation(self, base_points, perturbation_scale=0.5):
        """
        Generate points by perturbing a base configuration.
        
        Args:
            base_points: Base control points to perturb
            perturbation_scale: Scale of perturbation (μm)
            
        Returns:
            List of perturbed control point arrays
        """
        curves = []
        for base_curve in base_points:
            perturbed = []
            for x, y in base_curve:
                dx = np.random.normal(0, perturbation_scale)
                dy = np.random.normal(0, perturbation_scale)
                new_x = np.clip(x + dx, self.x_range[0], self.x_range[1])
                new_y = np.clip(y + dy, self.y_range[0], self.y_range[1])
                perturbed.append([float(new_x), float(new_y)])
            curves.append(perturbed)
        
        return curves
    
    def sample_latin_hypercube(self, n_samples):
        """
        Latin Hypercube Sampling for better coverage of design space.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of control point configurations
        """
        all_samples = []
        n_points = 5  # Fixed for LHS
        n_dims = n_points * 2  # x and y for each point
        
        # Generate LHS samples
        samples = np.zeros((n_samples, n_dims))
        for dim in range(n_dims):
            perm = np.random.permutation(n_samples)
            samples[:, dim] = (perm + np.random.random(n_samples)) / n_samples
        
        # Scale to ranges
        for i in range(n_samples):
            points = []
            for j in range(n_points):
                x = self.x_range[0] + samples[i, j*2] * (self.x_range[1] - self.x_range[0])
                y = self.y_range[0] + samples[i, j*2+1] * (self.y_range[1] - self.y_range[0])
                points.append([float(x), float(y)])
            
            # Sort by x for proper curve
            points.sort(key=lambda p: p[0])
            all_samples.append([points])
        
        return all_samples


# =============================================================================
# Data Generation Pipeline
# =============================================================================

class DataGenerationPipeline:
    """
    Complete pipeline for generating STCMT training data.
    """
    
    def __init__(self, 
                 output_dir='stcmt_dataset',
                 n_angles=3,
                 save_interval=50,
                 verbose=True):
        """
        Args:
            output_dir: Directory to save generated data
            n_angles: Number of angles for band dispersion extraction
            save_interval: Save progress every N samples
            verbose: Print detailed progress
        """
        self.output_dir = output_dir
        self.n_angles = n_angles
        self.save_interval = save_interval
        self.verbose = verbose
        
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.sampler = NURBSSampler()
        self.dataset = []
        self.failed_samples = []
        
    def generate_single_sample(self, control_points, sample_id):
        """
        Generate a single training sample.
        
        Args:
            control_points: NURBS control points
            sample_id: Unique sample identifier
            
        Returns:
            Sample dictionary or None if failed
        """
        try:
            # Create simulation
            sim = STCMTSimulation(control_points, degree=2)
            
            # Extract STCMT parameters
            stcmt_params = sim.extract_stcmt_parameters(n_angles=self.n_angles)
            
            # Close simulation
            sim.close()
            
            # Create sample entry
            sample = {
                'sample_id': sample_id,
                'control_points': control_points,
                'stcmt_params': stcmt_params,
                'timestamp': datetime.now().isoformat()
            }
            
            return sample
            
        except Exception as e:
            if self.verbose:
                print(f"  Error on sample {sample_id}: {e}")
            return None
    
    def generate_dataset(self, 
                         n_samples, 
                         sampling_strategy='mixed',
                         resume_from=None):
        """
        Generate complete training dataset.
        
        Args:
            n_samples: Total number of samples to generate
            sampling_strategy: 'random', 'smooth', 'symmetric', 'lhs', or 'mixed'
            resume_from: Path to partial dataset to resume from
            
        Returns:
            Generated dataset
        """
        print("=" * 60)
        print("STCMT Training Data Generation")
        print("=" * 60)
        print(f"Target samples: {n_samples}")
        print(f"Sampling strategy: {sampling_strategy}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        # Resume from partial dataset if provided
        start_id = 0
        if resume_from and os.path.exists(resume_from):
            with open(resume_from, 'r') as f:
                self.dataset = json.load(f)
            start_id = len(self.dataset)
            print(f"Resuming from {start_id} samples")
        
        # Generate samples based on strategy
        if sampling_strategy == 'lhs':
            # Pre-generate all samples with LHS
            all_control_points = self.sampler.sample_latin_hypercube(n_samples)
        else:
            all_control_points = None
        
        # Main generation loop
        sim = None  # Reuse simulation object
        
        for i in tqdm(range(start_id, n_samples), desc="Generating samples"):
            # Get control points based on strategy
            if sampling_strategy == 'random':
                control_points = self.sampler.sample_random()
            elif sampling_strategy == 'smooth':
                control_points = self.sampler.sample_smooth_curve()
            elif sampling_strategy == 'symmetric':
                control_points = self.sampler.sample_symmetric()
            elif sampling_strategy == 'lhs':
                control_points = all_control_points[i]
            else:  # mixed
                strategy = np.random.choice(['random', 'smooth', 'symmetric'])
                if strategy == 'random':
                    control_points = self.sampler.sample_random()
                elif strategy == 'smooth':
                    control_points = self.sampler.sample_smooth_curve()
                else:
                    control_points = self.sampler.sample_symmetric()
            
            try:
                # Initialize or update simulation
                if sim is None:
                    sim = STCMTSimulation(control_points, degree=2)
                else:
                    sim.set_control_points(control_points)
                
                # Extract STCMT parameters
                stcmt_params = sim.extract_stcmt_parameters(n_angles=self.n_angles)
                
                # Create sample
                sample = {
                    'sample_id': i,
                    'control_points': control_points,
                    'stcmt_params': stcmt_params,
                    'sampling_strategy': sampling_strategy if sampling_strategy != 'mixed' else np.random.choice(['random', 'smooth', 'symmetric'])
                }
                
                self.dataset.append(sample)
                
            except Exception as e:
                if self.verbose:
                    print(f"\n  Error on sample {i}: {e}")
                self.failed_samples.append({
                    'sample_id': i,
                    'control_points': control_points,
                    'error': str(e)
                })
                continue
            
            # Save progress periodically
            if (i + 1) % self.save_interval == 0:
                self._save_progress()
                if self.verbose:
                    print(f"\n  Progress saved: {len(self.dataset)} samples")
        
        # Close simulation
        if sim:
            sim.close()
        
        # Save final dataset
        self._save_final()
        
        return self.dataset
    
    def _save_progress(self):
        """Save partial progress to file"""
        progress_path = os.path.join(
            self.output_dir, 
            f'dataset_partial_{self.timestamp}.json'
        )
        self._save_json(self.dataset, progress_path)
    
    def _save_final(self):
        """Save final dataset and statistics"""
        # Save main dataset
        final_path = os.path.join(
            self.output_dir, 
            f'stcmt_dataset_{self.timestamp}.json'
        )
        self._save_json(self.dataset, final_path)
        
        # Save failed samples for analysis
        if self.failed_samples:
            failed_path = os.path.join(
                self.output_dir, 
                f'failed_samples_{self.timestamp}.json'
            )
            self._save_json(self.failed_samples, failed_path)
        
        # Save statistics
        stats = self._compute_statistics()
        stats_path = os.path.join(
            self.output_dir, 
            f'dataset_stats_{self.timestamp}.json'
        )
        self._save_json(stats, stats_path)
        
        print("\n" + "=" * 60)
        print("Dataset Generation Complete!")
        print("=" * 60)
        print(f"Total samples: {len(self.dataset)}")
        print(f"Failed samples: {len(self.failed_samples)}")
        print(f"Success rate: {len(self.dataset) / (len(self.dataset) + len(self.failed_samples)) * 100:.1f}%")
        print(f"Dataset saved to: {final_path}")
        print("=" * 60)
    
    def _save_json(self, data, filepath):
        """Save data to JSON with numpy type handling"""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(convert(data), f, indent=2)
    
    def _compute_statistics(self):
        """Compute dataset statistics"""
        if not self.dataset:
            return {}
        
        stats = {
            'n_samples': len(self.dataset),
            'n_failed': len(self.failed_samples)
        }
        
        # Extract STCMT parameter distributions
        omega_0_list = []
        gamma_list = []
        q_list = []
        
        for sample in self.dataset:
            pm = sample.get('stcmt_params', {}).get('primary_mode', {})
            if pm:
                if 'omega_0' in pm:
                    omega_0_list.append(pm['omega_0'])
                if 'gamma_total' in pm:
                    gamma_list.append(pm['gamma_total'])
                if 'Q_factor' in pm:
                    q_list.append(pm['Q_factor'])
        
        if omega_0_list:
            stats['omega_0'] = {
                'mean': float(np.mean(omega_0_list)),
                'std': float(np.std(omega_0_list)),
                'min': float(np.min(omega_0_list)),
                'max': float(np.max(omega_0_list))
            }
        
        if gamma_list:
            stats['gamma_total'] = {
                'mean': float(np.mean(gamma_list)),
                'std': float(np.std(gamma_list)),
                'min': float(np.min(gamma_list)),
                'max': float(np.max(gamma_list))
            }
        
        if q_list:
            stats['Q_factor'] = {
                'mean': float(np.mean(q_list)),
                'std': float(np.std(q_list)),
                'min': float(np.min(q_list)),
                'max': float(np.max(q_list))
            }
        
        return stats


# =============================================================================
# Dataset Splitting and Preprocessing
# =============================================================================

def split_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset into train/validation/test sets.
    
    Args:
        dataset_path: Path to complete dataset JSON
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        
    Returns:
        Paths to split dataset files
    """
    np.random.seed(seed)
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_set = [dataset[i] for i in train_indices]
    val_set = [dataset[i] for i in val_indices]
    test_set = [dataset[i] for i in test_indices]
    
    # Save splits
    base_dir = os.path.dirname(dataset_path)
    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    train_path = os.path.join(base_dir, f'{base_name}_train.json')
    val_path = os.path.join(base_dir, f'{base_name}_val.json')
    test_path = os.path.join(base_dir, f'{base_name}_test.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_set, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_set, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_set, f, indent=2)
    
    print(f"Dataset split complete:")
    print(f"  Training: {len(train_set)} samples -> {train_path}")
    print(f"  Validation: {len(val_set)} samples -> {val_path}")
    print(f"  Testing: {len(test_set)} samples -> {test_path}")
    
    return train_path, val_path, test_path


# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate training data for STCMT surrogate model'
    )
    parser.add_argument(
        '--n_samples', type=int, default=100,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--output_dir', type=str, default='stcmt_dataset',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--strategy', type=str, default='mixed',
        choices=['random', 'smooth', 'symmetric', 'lhs', 'mixed'],
        help='Sampling strategy for control points'
    )
    parser.add_argument(
        '--n_angles', type=int, default=3,
        help='Number of angles for band dispersion'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to partial dataset to resume from'
    )
    parser.add_argument(
        '--split', action='store_true',
        help='Split dataset into train/val/test after generation'
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    pipeline = DataGenerationPipeline(
        output_dir=args.output_dir,
        n_angles=args.n_angles,
        save_interval=50,
        verbose=True
    )
    
    dataset = pipeline.generate_dataset(
        n_samples=args.n_samples,
        sampling_strategy=args.strategy,
        resume_from=args.resume
    )
    
    # Split if requested
    if args.split and dataset:
        dataset_path = os.path.join(
            args.output_dir, 
            f'stcmt_dataset_{pipeline.timestamp}.json'
        )
        split_dataset(dataset_path)


if __name__ == '__main__':
    # Quick test run
    print("=" * 60)
    print("STCMT Data Generation - Quick Test")
    print("=" * 60)
    
    # Test sampler
    sampler = NURBSSampler()
    
    print("\nTesting sampling strategies:")
    print("-" * 40)
    
    random_pts = sampler.sample_random()
    print(f"Random: {len(random_pts[0])} control points")
    
    smooth_pts = sampler.sample_smooth_curve()
    print(f"Smooth: {len(smooth_pts[0])} control points")
    
    symmetric_pts = sampler.sample_symmetric()
    print(f"Symmetric: {len(symmetric_pts[0])} control points")
    
    lhs_samples = sampler.sample_latin_hypercube(5)
    print(f"LHS: Generated {len(lhs_samples)} configurations")
    
    print("\n" + "=" * 60)
    print("To generate a full dataset, run:")
    print("  python generate_training_data.py --n_samples 1000 --strategy mixed --split")
    print("=" * 60)
