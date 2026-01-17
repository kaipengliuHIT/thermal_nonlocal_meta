"""
PyTorch Surrogate Model for NURBS to STCMT Parameter Mapping

This module implements an end-to-end deep learning surrogate model that maps
NURBS control point parameters directly to Spatiotemporal Coupled Mode Theory
(STCMT) parameters, bypassing expensive full-wave FDTD simulations.

Based on: "Inverse Design and Experimental Verification of Nonlocal Thermal Metasurfaces
via NURBS and Spatiotemporal Coupled-Mode Theory"

Architecture:
- Encoder: Maps NURBS control points to latent representation
- Decoder: Maps latent representation to STCMT parameters
- Physics-informed: Incorporates STCMT constraints in loss function

Key STCMT outputs:
- Resonance frequencies (ω₀)
- Radiative/non-radiative decay rates (γ_rad, γ_nrad)
- Inter-modal coupling coefficients (κ)
- Effective mass / band curvature parameters
- Coupling length for nonlocality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


# =============================================================================
# STCMT Physics Module
# =============================================================================

class STCMTPhysics:
    """
    Spatiotemporal Coupled Mode Theory physics calculations.
    
    STCMT describes the optical response of nonlocal metasurfaces through
    mode coupling and spatiotemporal dynamics.
    """
    
    @staticmethod
    def compute_reflection(omega, omega_0, gamma_rad, gamma_nrad, background=0):
        """
        Compute reflection coefficient using temporal CMT.
        
        r(ω) = r_bg - (2γ_rad) / (i(ω - ω₀) + γ_rad + γ_nrad)
        
        Args:
            omega: Angular frequency (can be tensor)
            omega_0: Resonance frequency
            gamma_rad: Radiative decay rate
            gamma_nrad: Non-radiative decay rate
            background: Background reflection
            
        Returns:
            Complex reflection coefficient
        """
        denominator = 1j * (omega - omega_0) + gamma_rad + gamma_nrad
        r = background - 2 * gamma_rad / denominator
        return r
    
    @staticmethod
    def compute_absorption(omega, omega_0, gamma_rad, gamma_nrad):
        """
        Compute absorption using TCMT.
        
        A(ω) = 4γ_rad * γ_nrad / ((ω - ω₀)² + (γ_rad + γ_nrad)²)
        
        Args:
            omega: Angular frequency
            omega_0: Resonance frequency
            gamma_rad: Radiative decay rate
            gamma_nrad: Non-radiative decay rate
            
        Returns:
            Absorption spectrum
        """
        numerator = 4 * gamma_rad * gamma_nrad
        denominator = (omega - omega_0)**2 + (gamma_rad + gamma_nrad)**2
        return numerator / denominator
    
    @staticmethod
    def compute_nonlocal_response(omega, k_parallel, omega_0, gamma, effective_mass, c=3e8):
        """
        Compute nonlocal response with k-dependent dispersion.
        
        ω(k) = ω₀ + ℏk²/(2m*)
        
        The nonlocal response accounts for spatial dispersion through
        the effective mass parameter.
        
        Args:
            omega: Angular frequency
            k_parallel: Parallel wavevector
            omega_0: Resonance frequency at k=0
            gamma: Total decay rate
            effective_mass: Effective mass parameter
            c: Speed of light
            
        Returns:
            Nonlocal optical response
        """
        hbar = 1.054e-34
        
        # k-dependent resonance frequency
        omega_k = omega_0 + hbar * k_parallel**2 / (2 * effective_mass)
        
        # Lorentzian response at shifted frequency
        response = gamma**2 / ((omega - omega_k)**2 + gamma**2)
        
        return response
    
    @staticmethod
    def coupling_length(band_curvature, gamma):
        """
        Calculate nonlocality coupling length.
        
        L_c ∝ 1/κ ∝ |∂²ω/∂k²| / γ
        
        Args:
            band_curvature: Second derivative of dispersion
            gamma: Decay rate
            
        Returns:
            Coupling length
        """
        if gamma == 0:
            return float('inf')
        return abs(band_curvature) / gamma


# =============================================================================
# Neural Network Architectures
# =============================================================================

class NURBSEncoder(nn.Module):
    """
    Encoder network for NURBS control points.
    
    Takes variable-length control point sequences and produces
    fixed-size latent representation using attention mechanism.
    """
    
    def __init__(self, input_dim=2, hidden_dim=128, latent_dim=64, 
                 n_heads=4, n_layers=3, max_points=20):
        """
        Args:
            input_dim: Dimension of each control point (2 for x,y)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            max_points: Maximum number of control points
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_points = max_points
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_points, hidden_dim) * 0.02
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Global pooling and projection to latent space
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, control_points, mask=None):
        """
        Args:
            control_points: (batch, n_points, 2) tensor
            mask: Optional padding mask
            
        Returns:
            latent: (batch, latent_dim) latent representation
        """
        batch_size, n_points, _ = control_points.shape
        
        # Project input
        x = self.input_proj(control_points)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :n_points, :]
        
        # Transformer encoding
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # Global pooling (mean over sequence)
        if mask is not None:
            # Masked mean pooling
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        # Project to latent space
        latent = self.global_pool(x)
        
        return latent


class STCMTDecoder(nn.Module):
    """
    Decoder network for STCMT parameters.
    
    Maps latent representation to physical STCMT parameters with
    appropriate constraints (e.g., positive decay rates).
    """
    
    def __init__(self, latent_dim=64, hidden_dim=128, n_layers=3):
        """
        Args:
            latent_dim: Input latent dimension
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Build MLP layers
        layers = []
        in_dim = latent_dim
        
        for i in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output heads for different STCMT parameters
        # All outputs use softplus to ensure positivity where needed
        
        # Resonance frequency (normalized to center wavelength)
        self.omega_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Decay rates (must be positive)
        self.gamma_rad_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        self.gamma_nrad_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # Band curvature (can be positive or negative)
        self.curvature_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Effective mass (positive)
        self.mass_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # Coupling coefficients
        self.coupling_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 4)  # Multiple coupling terms
        )
        
        # Q-factor prediction (auxiliary output)
        self.q_factor_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
    def forward(self, latent):
        """
        Args:
            latent: (batch, latent_dim) latent representation
            
        Returns:
            stcmt_params: Dictionary of STCMT parameters
        """
        # Shared feature extraction
        features = self.mlp(latent)
        
        # Compute individual parameters
        omega_0 = self.omega_head(features)
        gamma_rad = self.gamma_rad_head(features)
        gamma_nrad = self.gamma_nrad_head(features)
        band_curvature = self.curvature_head(features)
        effective_mass = self.mass_head(features)
        coupling_coeffs = self.coupling_head(features)
        q_factor = self.q_factor_head(features)
        
        return {
            'omega_0': omega_0,
            'gamma_rad': gamma_rad,
            'gamma_nrad': gamma_nrad,
            'gamma_total': gamma_rad + gamma_nrad,
            'band_curvature': band_curvature,
            'effective_mass': effective_mass,
            'coupling_coeffs': coupling_coeffs,
            'Q_factor': q_factor
        }


class SpectrumPredictor(nn.Module):
    """
    Predicts optical spectrum from STCMT parameters.
    
    Uses STCMT physics to generate absorption/reflection spectra
    from the decoded parameters, enabling end-to-end training.
    """
    
    def __init__(self, n_wavelengths=201, wavelength_range=(11.6e-6, 13.6e-6)):
        """
        Args:
            n_wavelengths: Number of wavelength points
            wavelength_range: (min, max) wavelength in meters
        """
        super().__init__()
        
        self.n_wavelengths = n_wavelengths
        self.c = 3e8  # Speed of light
        
        # Fixed wavelength grid
        wavelengths = torch.linspace(
            wavelength_range[0], 
            wavelength_range[1], 
            n_wavelengths
        )
        self.register_buffer('wavelengths', wavelengths)
        self.register_buffer('omega', 2 * np.pi * self.c / wavelengths)
        
        # Normalization parameters for omega_0
        self.omega_center = 2 * np.pi * self.c / 12.6e-6
        self.omega_scale = self.omega_center * 0.1
        
    def forward(self, stcmt_params):
        """
        Args:
            stcmt_params: Dictionary from STCMTDecoder
            
        Returns:
            spectra: Dictionary with absorption and reflection spectra
        """
        # Extract and denormalize parameters
        omega_0 = stcmt_params['omega_0'] * self.omega_scale + self.omega_center
        gamma_rad = stcmt_params['gamma_rad'] * 1e12  # Scale to physical units
        gamma_nrad = stcmt_params['gamma_nrad'] * 1e12
        
        batch_size = omega_0.shape[0]
        
        # Expand omega grid for batch computation
        omega = self.omega.unsqueeze(0).expand(batch_size, -1)  # (batch, n_wl)
        omega_0 = omega_0.expand(-1, self.n_wavelengths)  # (batch, n_wl)
        gamma_rad = gamma_rad.expand(-1, self.n_wavelengths)
        gamma_nrad = gamma_nrad.expand(-1, self.n_wavelengths)
        
        # Compute absorption using TCMT formula
        numerator = 4 * gamma_rad * gamma_nrad
        denominator = (omega - omega_0)**2 + (gamma_rad + gamma_nrad)**2
        absorption = numerator / (denominator + 1e-20)
        
        # Compute reflection (simplified, assuming normal incidence)
        reflection = 1 - absorption
        
        return {
            'absorption': absorption,
            'reflection': reflection,
            'wavelengths': self.wavelengths
        }


class NURBStoSTCMTSurrogate(nn.Module):
    """
    Complete end-to-end surrogate model: NURBS → STCMT → Spectrum
    
    This model provides a differentiable mapping from NURBS control points
    to optical spectra via STCMT parameters, enabling gradient-based
    inverse design without adjoint methods.
    """
    
    def __init__(self, 
                 input_dim=2,
                 hidden_dim=128,
                 latent_dim=64,
                 n_encoder_layers=3,
                 n_decoder_layers=3,
                 n_wavelengths=201,
                 wavelength_range=(11.6e-6, 13.6e-6)):
        """
        Args:
            input_dim: Control point dimension (2 for x,y)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            n_encoder_layers: Number of encoder transformer layers
            n_decoder_layers: Number of decoder MLP layers
            n_wavelengths: Number of wavelength points for spectrum
            wavelength_range: Wavelength range for spectrum prediction
        """
        super().__init__()
        
        self.encoder = NURBSEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=n_encoder_layers
        )
        
        self.decoder = STCMTDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_decoder_layers
        )
        
        self.spectrum_predictor = SpectrumPredictor(
            n_wavelengths=n_wavelengths,
            wavelength_range=wavelength_range
        )
        
    def forward(self, control_points, mask=None, return_stcmt=True):
        """
        Args:
            control_points: (batch, n_points, 2) NURBS control points
            mask: Optional padding mask
            return_stcmt: Whether to return STCMT parameters
            
        Returns:
            If return_stcmt:
                (spectra, stcmt_params, latent)
            Else:
                spectra
        """
        # Encode NURBS to latent
        latent = self.encoder(control_points, mask)
        
        # Decode to STCMT parameters
        stcmt_params = self.decoder(latent)
        
        # Predict spectrum
        spectra = self.spectrum_predictor(stcmt_params)
        
        if return_stcmt:
            return spectra, stcmt_params, latent
        else:
            return spectra
    
    def predict_stcmt(self, control_points, mask=None):
        """
        Predict only STCMT parameters (faster inference).
        """
        latent = self.encoder(control_points, mask)
        stcmt_params = self.decoder(latent)
        return stcmt_params


# =============================================================================
# Dataset and Data Loading
# =============================================================================

class STCMTDataset(Dataset):
    """
    Dataset for NURBS → STCMT surrogate model training.
    """
    
    def __init__(self, data_path, max_points=20, normalize=True):
        """
        Args:
            data_path: Path to JSON dataset file
            max_points: Maximum control points (for padding)
            normalize: Whether to normalize inputs/outputs
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.max_points = max_points
        self.normalize = normalize
        
        # Compute normalization statistics
        if normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization"""
        all_points = []
        all_omega = []
        all_gamma = []
        
        for sample in self.data:
            for curve in sample['control_points']:
                all_points.extend(curve)
            
            if 'primary_mode' in sample.get('stcmt_params', {}):
                pm = sample['stcmt_params']['primary_mode']
                all_omega.append(pm.get('omega_0', 0))
                all_gamma.append(pm.get('gamma_total', 0))
        
        all_points = np.array(all_points)
        self.point_mean = all_points.mean(axis=0)
        self.point_std = all_points.std(axis=0) + 1e-6
        
        self.omega_mean = np.mean(all_omega) if all_omega else 1.45e14
        self.omega_std = np.std(all_omega) if all_omega else 1e13
        self.gamma_mean = np.mean(all_gamma) if all_gamma else 1e12
        self.gamma_std = np.std(all_gamma) if all_gamma else 1e11
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get control points (flatten all curves)
        control_points = []
        for curve in sample['control_points']:
            control_points.extend(curve)
        
        control_points = np.array(control_points, dtype=np.float32)
        
        # Normalize if requested
        if self.normalize:
            control_points = (control_points - self.point_mean) / self.point_std
        
        # Pad to max_points
        n_points = len(control_points)
        if n_points < self.max_points:
            padding = np.zeros((self.max_points - n_points, 2), dtype=np.float32)
            control_points = np.vstack([control_points, padding])
            mask = np.array([False] * n_points + [True] * (self.max_points - n_points))
        else:
            control_points = control_points[:self.max_points]
            mask = np.array([False] * self.max_points)
        
        # Extract STCMT targets
        stcmt = sample.get('stcmt_params', {})
        pm = stcmt.get('primary_mode', {})
        nl = stcmt.get('nonlocality', {})
        
        targets = {
            'omega_0': (pm.get('omega_0', self.omega_mean) - self.omega_mean) / self.omega_std if self.normalize else pm.get('omega_0', 0),
            'gamma_rad': pm.get('gamma_rad', self.gamma_mean) / self.gamma_std if self.normalize else pm.get('gamma_rad', 0),
            'gamma_nrad': pm.get('gamma_nrad', self.gamma_mean) / self.gamma_std if self.normalize else pm.get('gamma_nrad', 0),
            'Q_factor': pm.get('Q_factor', 100) / 1000,  # Normalize Q-factor
            'band_curvature': nl.get('band_curvature', 0),
        }
        
        # Get spectrum if available
        normal = stcmt.get('normal_incidence', {})
        if 'absorptance' in normal:
            spectrum = np.array(normal['absorptance'], dtype=np.float32)
        else:
            spectrum = np.zeros(201, dtype=np.float32)
        
        return {
            'control_points': torch.from_numpy(control_points),
            'mask': torch.from_numpy(mask),
            'targets': {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()},
            'spectrum': torch.from_numpy(spectrum)
        }


# =============================================================================
# Training Functions
# =============================================================================

class STCMTLoss(nn.Module):
    """
    Physics-informed loss function for STCMT surrogate model.
    
    Combines:
    - MSE loss for STCMT parameter prediction
    - Spectrum reconstruction loss
    - Physics consistency regularization
    """
    
    def __init__(self, 
                 param_weight=1.0,
                 spectrum_weight=1.0,
                 physics_weight=0.1):
        super().__init__()
        
        self.param_weight = param_weight
        self.spectrum_weight = spectrum_weight
        self.physics_weight = physics_weight
        
    def forward(self, predictions, targets, stcmt_params):
        """
        Args:
            predictions: Predicted spectra dictionary
            targets: Target dictionary with STCMT params and spectrum
            stcmt_params: Predicted STCMT parameters
            
        Returns:
            Total loss and component losses
        """
        losses = {}
        
        # STCMT parameter losses
        if 'omega_0' in targets:
            losses['omega_0'] = F.mse_loss(
                stcmt_params['omega_0'].squeeze(), 
                targets['omega_0']
            )
        
        if 'gamma_rad' in targets:
            losses['gamma_rad'] = F.mse_loss(
                stcmt_params['gamma_rad'].squeeze(),
                targets['gamma_rad']
            )
            
        if 'gamma_nrad' in targets:
            losses['gamma_nrad'] = F.mse_loss(
                stcmt_params['gamma_nrad'].squeeze(),
                targets['gamma_nrad']
            )
            
        if 'Q_factor' in targets:
            losses['Q_factor'] = F.mse_loss(
                stcmt_params['Q_factor'].squeeze(),
                targets['Q_factor']
            )
        
        # Spectrum reconstruction loss
        if 'spectrum' in targets and predictions is not None:
            pred_absorption = predictions['absorption']
            target_spectrum = targets['spectrum']
            
            # Ensure same length
            min_len = min(pred_absorption.shape[-1], target_spectrum.shape[-1])
            losses['spectrum'] = F.mse_loss(
                pred_absorption[..., :min_len],
                target_spectrum[..., :min_len]
            )
        
        # Physics consistency: Q = ω₀ / (2γ_total)
        if 'omega_0' in stcmt_params and 'gamma_total' in stcmt_params:
            q_computed = stcmt_params['omega_0'] / (2 * stcmt_params['gamma_total'] + 1e-10)
            q_predicted = stcmt_params['Q_factor']
            losses['physics_q'] = F.mse_loss(q_computed, q_predicted)
        
        # Combine losses
        total_loss = 0
        for key, loss in losses.items():
            if 'spectrum' in key:
                total_loss += self.spectrum_weight * loss
            elif 'physics' in key:
                total_loss += self.physics_weight * loss
            else:
                total_loss += self.param_weight * loss
        
        return total_loss, losses


def train_surrogate_model(model, train_loader, val_loader, 
                          n_epochs=100, lr=1e-4, device='cuda',
                          save_dir='surrogate_checkpoints'):
    """
    Train the NURBS to STCMT surrogate model.
    
    Args:
        model: NURBStoSTCMTSurrogate model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Training device
        save_dir: Directory for saving checkpoints
        
    Returns:
        Trained model and training history
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = STCMTLoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'component_losses': []
    }
    
    best_val_loss = float('inf')
    
    print("=" * 60)
    print("Training NURBS → STCMT Surrogate Model")
    print("=" * 60)
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            control_points = batch['control_points'].to(device)
            mask = batch['mask'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            spectrum = batch['spectrum'].to(device)
            targets['spectrum'] = spectrum
            
            optimizer.zero_grad()
            
            # Forward pass
            spectra, stcmt_params, latent = model(control_points, mask)
            
            # Compute loss
            loss, component_losses = criterion(spectra, targets, stcmt_params)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                control_points = batch['control_points'].to(device)
                mask = batch['mask'].to(device)
                targets = {k: v.to(device) for k, v in batch['targets'].items()}
                spectrum = batch['spectrum'].to(device)
                targets['spectrum'] = spectrum
                
                spectra, stcmt_params, latent = model(control_points, mask)
                loss, _ = criterion(spectra, targets, stcmt_params)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Print progress
        print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.6f})")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, os.path.join(save_dir, 'final_model.pt'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300)
    plt.close()
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 60)
    
    return model, history


# =============================================================================
# Inverse Design Using Surrogate Model
# =============================================================================

def inverse_design(model, target_spectrum, n_iterations=1000, lr=0.1, device='cuda'):
    """
    Perform gradient-based inverse design using the surrogate model.
    
    Optimizes NURBS control points to achieve target spectrum.
    
    Args:
        model: Trained NURBStoSTCMTSurrogate model
        target_spectrum: Target absorption spectrum
        n_iterations: Number of optimization iterations
        lr: Learning rate for control point optimization
        device: Computation device
        
    Returns:
        Optimized control points and final spectrum
    """
    model.eval()
    model = model.to(device)
    
    # Initialize control points (learnable)
    # Start with a simple curved shape
    n_points = 6
    init_points = torch.tensor([
        [-4.0, -3.0], [-2.0, 0.0], [0.0, 2.0],
        [2.0, 0.0], [4.0, -2.0], [5.0, 1.0]
    ], dtype=torch.float32, device=device)
    
    control_points = nn.Parameter(init_points.unsqueeze(0))  # (1, n_points, 2)
    
    target_spectrum = torch.tensor(target_spectrum, dtype=torch.float32, device=device)
    if target_spectrum.dim() == 1:
        target_spectrum = target_spectrum.unsqueeze(0)
    
    optimizer = torch.optim.Adam([control_points], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iterations)
    
    history = {'loss': [], 'control_points': []}
    
    print("Starting inverse design optimization...")
    
    for i in tqdm(range(n_iterations)):
        optimizer.zero_grad()
        
        # Forward pass through surrogate
        spectra, stcmt_params, _ = model(control_points)
        pred_spectrum = spectra['absorption']
        
        # Compute loss
        min_len = min(pred_spectrum.shape[-1], target_spectrum.shape[-1])
        loss = F.mse_loss(pred_spectrum[..., :min_len], target_spectrum[..., :min_len])
        
        # Add regularization for smooth control points
        point_diffs = control_points[:, 1:, :] - control_points[:, :-1, :]
        smoothness_loss = 0.01 * (point_diffs ** 2).mean()
        
        total_loss = loss + smoothness_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record history
        history['loss'].append(total_loss.item())
        
        if i % 100 == 0:
            print(f"  Iteration {i}: Loss = {total_loss.item():.6f}")
    
    # Get final result
    with torch.no_grad():
        final_spectra, final_stcmt, _ = model(control_points)
    
    result = {
        'control_points': control_points.detach().cpu().numpy()[0],
        'predicted_spectrum': final_spectra['absorption'].detach().cpu().numpy()[0],
        'stcmt_params': {k: v.detach().cpu().numpy() for k, v in final_stcmt.items()},
        'optimization_history': history
    }
    
    print(f"Inverse design complete. Final loss: {history['loss'][-1]:.6f}")
    
    return result


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("NURBS → STCMT Surrogate Model")
    print("=" * 60)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = NURBStoSTCMTSurrogate(
        input_dim=2,
        hidden_dim=128,
        latent_dim=64,
        n_encoder_layers=3,
        n_decoder_layers=3,
        n_wavelengths=201,
        wavelength_range=(11.6e-6, 13.6e-6)
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_points = torch.randn(2, 6, 2)  # Batch of 2, 6 control points each
    
    spectra, stcmt_params, latent = model(test_points)
    
    print(f"Input shape: {test_points.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Absorption spectrum shape: {spectra['absorption'].shape}")
    print("\nSTCMT parameters:")
    for k, v in stcmt_params.items():
        print(f"  {k}: {v.shape}")
    
    print("\n" + "=" * 60)
    print("Model ready for training!")
    print("=" * 60)
    print("\nTo train the model, use:")
    print("  from surrogate_model import train_surrogate_model, STCMTDataset")
    print("  dataset = STCMTDataset('stcmt_dataset/stcmt_dataset_XXXXX.json')")
    print("  train_loader = DataLoader(dataset, batch_size=32, shuffle=True)")
    print("  model, history = train_surrogate_model(model, train_loader, val_loader)")
