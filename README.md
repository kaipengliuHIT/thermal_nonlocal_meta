# Nonlocal Thermal Metasurface Inverse Design

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Lumerical](https://img.shields.io/badge/Lumerical-FDTD-orange.svg)](https://www.lumerical.com/)

This repository contains the code for **"Inverse Design and Experimental Verification of Nonlocal Thermal Metasurfaces via NURBS and Spatiotemporal Coupled-Mode Theory"**.

## Overview

We present a two-stage inverse design framework for nonlocal thermal metasurfaces:

1. **Stage 1: PSO-based Local Optimization** - Particle Swarm Optimization for fine-tuning perturbations to BIC-supporting metasurface structures
2. **Stage 2: Deep Learning Surrogate Model** - End-to-end differentiable model mapping NURBS control points to STCMT parameters for global inverse design

### Key Features

- **NURBS Parameterization**: Flexible representation of meta-atom geometry using Non-Uniform Rational B-Splines
- **STCMT Integration**: Spatiotemporal Coupled Mode Theory for efficient modeling of nonlocal optical response
- **Physics-Informed Learning**: Surrogate model incorporates physical constraints from STCMT
- **Lumerical FDTD Integration**: Direct interface with commercial electromagnetic simulation software

## Repository Structure

```
thermal_nonlocal_meta/
├── README.md                      # This file
├── ZnS.txt                        # ZnS optical constants (refractive index data)
├── sim_for_pso.py                 # Lumerical simulation interface for PSO optimization
├── sim_for_dl.py                  # Lumerical simulation interface for deep learning
├── pso_optimization.py            # PSO optimizer implementation
├── sim_stcmt_extraction.py        # Single-period simulation for STCMT parameter extraction
├── surrogate_model.py             # PyTorch surrogate model (NURBS → STCMT)
└── generate_training_data.py      # Training data generation pipeline
```

## Installation

### Prerequisites

- Python 3.8+
- Lumerical FDTD (v2024 or later)
- CUDA-capable GPU (recommended for training)

### Dependencies

```bash
pip install numpy scipy matplotlib torch tqdm
```

### Lumerical API Setup

Ensure the Lumerical Python API is accessible. Update the path in the simulation scripts if needed:

```python
lumapi = importlib.machinery.SourceFileLoader(
    'lumapi', 
    'C:\\Program Files\\Lumerical\\v241\\api\\python\\lumapi.py'
).load_module()
```

## Usage

### Stage 1: PSO Optimization

Run the PSO optimizer to fine-tune metasurface perturbations:

```python
from pso_optimization import PSOConfig, PSOOptimizer

# Configure PSO
config = PSOConfig(
    n_particles=30,
    n_iterations=100,
    target_wavelength=12.6e-6  # Target resonance wavelength
)

# Run optimization
optimizer = PSOOptimizer(config)
best_solution, best_fitness = optimizer.optimize()
```

### Stage 2: Deep Learning Inverse Design

#### Step 1: Generate Training Data

```bash
python generate_training_data.py --n_samples 1000 --strategy mixed --split
```

This runs single-period FDTD simulations to extract STCMT parameters for various NURBS configurations.

#### Step 2: Train Surrogate Model

```python
from surrogate_model import NURBStoSTCMTSurrogate, STCMTDataset, train_surrogate_model
from torch.utils.data import DataLoader

# Load datasets
train_set = STCMTDataset('stcmt_dataset/stcmt_dataset_train.json')
val_set = STCMTDataset('stcmt_dataset/stcmt_dataset_val.json')

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# Train model
model = NURBStoSTCMTSurrogate()
train_surrogate_model(model, train_loader, val_loader, n_epochs=100)
```

#### Step 3: Inverse Design

```python
from surrogate_model import inverse_design, NURBStoSTCMTSurrogate
import torch

# Load trained model
model = NURBStoSTCMTSurrogate()
checkpoint = torch.load('surrogate_checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Define target spectrum (asymmetric emissivity)
target_spectrum = create_target_spectrum()

# Run gradient-based inverse design
result = inverse_design(
    model, 
    target_spectrum, 
    n_iterations=1000,
    learning_rate=0.01
)

print("Optimized control points:", result['control_points'])
```

## STCMT Parameters

The surrogate model predicts the following Spatiotemporal Coupled Mode Theory parameters:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Resonance frequency | ω₀ | Central frequency of the resonance |
| Radiative decay rate | γ_rad | Coupling rate to free-space radiation |
| Non-radiative decay rate | γ_nrad | Intrinsic absorption losses |
| Q-factor | Q | Quality factor of the resonance |
| Effective mass | m* | Band curvature parameter |
| Coupling length | L_c | Nonlocality coupling length |

## Metasurface Structure

The designed metasurface consists of:
- **Top layer**: Ge (Germanium) - 100 nm
- **Middle layer**: ZnS (Zinc Sulfide) - 600 nm  
- **Bottom layer**: Ge (Germanium) - 100 nm
- **Reflector**: Au (Gold) - 50 nm
- **Patterned layer**: Ag (Silver) NURBS-defined structures

## Citation

If you use this code in your research, please cite:

```bibtex
@article{liu2025nonlocal,
  title={Inverse Design and Experimental Verification of Nonlocal Thermal Metasurfaces via NURBS and Spatiotemporal Coupled-Mode Theory},
  author={Liu, Kaipeng and others},
  journal={TBD},
  year={2025}
}
```

## Contact

For questions or collaborations, please contact the authors or open an issue on GitHub.
