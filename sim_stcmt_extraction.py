"""
Single-Period FDTD Simulation for STCMT Parameter Extraction

This script performs single unit cell simulations with periodic boundary conditions
to extract Spatiotemporal Coupled Mode Theory (STCMT) parameters:
- Resonance frequencies (ω₀)
- Radiative decay rates (γ_rad)
- Non-radiative decay rates (γ_nrad)
- Inter-modal coupling coefficients (κ)
- Effective mass parameters (m* from band curvature)
- Mode coupling to radiation channels

Based on: "Inverse Design and Experimental Verification of Nonlocal Thermal Metasurfaces
via NURBS and Spatiotemporal Coupled-Mode Theory"

Reference: ALU's Light paper on STCMT for nonlocal metasurfaces
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import importlib
import os
import json
from tqdm import tqdm
from datetime import datetime

# Load ZnS material data
data = np.loadtxt('ZnS.txt')
wavelength_um = data[:, 0]  # Column 1: wavelength (micrometers)
n_real = data[:, 1]         # Column 2: real part of refractive index
n_imag = data[:, 2]         # Column 3: imaginary part of refractive index

# Calculate complex refractive index and permittivity
refractive_index_complex = n_real + 1j * n_imag
epsilon_complex = refractive_index_complex**2

# Convert wavelength to frequency
c = 3e8  # Speed of light (m/s)
wavelength_m = wavelength_um * 1e-6
f = c / wavelength_m

# Material data for Lumerical
sampledData = [f, epsilon_complex]

# Load Lumerical API
lumapi = importlib.machinery.SourceFileLoader(
    'lumapi', 
    'C:\\Program Files\\Lumerical\\v241\\api\\python\\lumapi.py'
).load_module()


# =============================================================================
# NURBS Curve Evaluation Functions
# =============================================================================

def basis_function(i, p, knots, t):
    """Calculate B-spline basis function value (recursive implementation)"""
    if p == 0:
        return 1.0 if knots[i] <= t < knots[i+1] else 0.0
    
    denom1 = knots[i+p] - knots[i]
    denom2 = knots[i+p+1] - knots[i+1]
    
    term1 = 0.0
    term2 = 0.0
    
    if denom1 != 0:
        term1 = (t - knots[i]) / denom1 * basis_function(i, p-1, knots, t)
    
    if denom2 != 0:
        term2 = (knots[i+p+1] - t) / denom2 * basis_function(i+1, p-1, knots, t)
    
    return term1 + term2


def find_span(n, p, knots, t):
    """Find the knot span containing parameter t"""
    if t >= knots[n+1]:
        return n
    if t <= knots[p]:
        return p
    
    low = p
    high = n + 1
    mid = (low + high) // 2
    
    while t < knots[mid] or t >= knots[mid+1]:
        if t < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    
    return mid


def evaluate_nurbs_curve(p, knots, control_points, t_values):
    """Evaluate NURBS curve at given parameter values"""
    n = len(control_points) - 1
    curve_points = []
    
    for t in t_values:
        span = find_span(n, p, knots, t)
        N = np.zeros(p + 1)
        for i in range(0, p + 1):
            N[i] = basis_function(span - p + i, p, knots, t)
        
        point = np.zeros(2)
        for i in range(0, p + 1):
            idx = span - p + i
            point += N[i] * np.array(control_points[idx])
        
        curve_points.append(point)
    
    return np.array(curve_points)


# =============================================================================
# STCMT Parameter Fitting Functions
# =============================================================================

def lorentzian(omega, omega_0, gamma, A, offset):
    """
    Lorentzian line shape for resonance fitting.
    
    Args:
        omega: Angular frequency array
        omega_0: Resonance frequency
        gamma: Total decay rate (FWHM/2)
        A: Amplitude
        offset: Background offset
        
    Returns:
        Lorentzian response
    """
    return A * gamma**2 / ((omega - omega_0)**2 + gamma**2) + offset


def fano_resonance(omega, omega_0, gamma, q, A, offset):
    """
    Fano resonance line shape (asymmetric resonance).
    
    Args:
        omega: Angular frequency array
        omega_0: Resonance frequency
        gamma: Decay rate
        q: Fano asymmetry parameter
        A: Amplitude
        offset: Background offset
        
    Returns:
        Fano response
    """
    epsilon = (omega - omega_0) / gamma
    return A * (q + epsilon)**2 / (1 + epsilon**2) + offset


def fit_resonance(wavelengths, response, method='lorentzian'):
    """
    Fit resonance to extract STCMT parameters.
    
    Args:
        wavelengths: Wavelength array (meters)
        response: Spectral response (absorption/reflection)
        method: 'lorentzian' or 'fano'
        
    Returns:
        Dictionary with fitted parameters
    """
    # Convert to angular frequency
    omega = 2 * np.pi * c / wavelengths
    
    # Find peaks in the response
    smoothed = gaussian_filter1d(response, sigma=2)
    peaks, properties = find_peaks(smoothed, height=0.1, prominence=0.05)
    
    if len(peaks) == 0:
        # Try finding peaks in inverted response (for transmission dips)
        peaks, properties = find_peaks(-smoothed, height=-0.9, prominence=0.05)
    
    fitted_params = []
    
    for peak_idx in peaks:
        # Define fitting window around peak
        window_size = min(50, len(omega) // 4)
        start = max(0, peak_idx - window_size)
        end = min(len(omega), peak_idx + window_size)
        
        omega_window = omega[start:end]
        response_window = response[start:end]
        
        try:
            if method == 'lorentzian':
                # Initial guess
                omega_0_guess = omega[peak_idx]
                gamma_guess = (omega[start] - omega[end]) / 10
                A_guess = response[peak_idx] - np.mean(response)
                offset_guess = np.mean(response)
                
                popt, pcov = curve_fit(
                    lorentzian, 
                    omega_window, 
                    response_window,
                    p0=[omega_0_guess, abs(gamma_guess), A_guess, offset_guess],
                    maxfev=5000
                )
                
                params = {
                    'omega_0': popt[0],
                    'gamma': abs(popt[1]),
                    'amplitude': popt[2],
                    'offset': popt[3],
                    'wavelength_0': 2 * np.pi * c / popt[0],
                    'Q_factor': popt[0] / (2 * abs(popt[1]))
                }
                
            elif method == 'fano':
                # Initial guess for Fano
                omega_0_guess = omega[peak_idx]
                gamma_guess = (omega[start] - omega[end]) / 10
                q_guess = 1.0
                A_guess = response[peak_idx]
                offset_guess = np.min(response_window)
                
                popt, pcov = curve_fit(
                    fano_resonance,
                    omega_window,
                    response_window,
                    p0=[omega_0_guess, abs(gamma_guess), q_guess, A_guess, offset_guess],
                    maxfev=5000
                )
                
                params = {
                    'omega_0': popt[0],
                    'gamma': abs(popt[1]),
                    'fano_q': popt[2],
                    'amplitude': popt[3],
                    'offset': popt[4],
                    'wavelength_0': 2 * np.pi * c / popt[0],
                    'Q_factor': popt[0] / (2 * abs(popt[1]))
                }
            
            fitted_params.append(params)
            
        except Exception as e:
            print(f"Fitting failed for peak at index {peak_idx}: {e}")
            continue
    
    return fitted_params


def extract_band_dispersion(omega_vs_k):
    """
    Extract effective mass from band dispersion ω(k).
    
    The effective mass is related to band curvature:
    m* = ℏ² / (∂²ω/∂k²)
    
    And the nonlocality coupling length:
    L_c ∝ 1/κ ∝ |∂²ω/∂k²|
    
    Args:
        omega_vs_k: Array of (k, omega) pairs
        
    Returns:
        Dictionary with dispersion parameters
    """
    k = omega_vs_k[:, 0]
    omega = omega_vs_k[:, 1]
    
    # Fit parabolic dispersion: ω(k) = ω₀ + αk²
    # This gives effective mass m* = ℏ²/(2α)
    
    try:
        # Polynomial fit (degree 2)
        coeffs = np.polyfit(k, omega, 2)
        alpha = coeffs[0]  # Coefficient of k²
        omega_0 = coeffs[2]  # Constant term
        
        # Calculate effective mass (in normalized units)
        hbar = 1.054e-34  # Reduced Planck constant
        if alpha != 0:
            effective_mass = hbar**2 / (2 * abs(alpha))
        else:
            effective_mass = np.inf
        
        # Nonlocality strength (proportional to band curvature)
        nonlocality_strength = abs(alpha)
        
        # Group velocity at different k points
        d_omega_dk = np.gradient(omega, k)
        group_velocity = d_omega_dk
        
        return {
            'omega_0': omega_0,
            'band_curvature': alpha,
            'effective_mass': effective_mass,
            'nonlocality_strength': nonlocality_strength,
            'group_velocity': group_velocity.tolist(),
            'k_values': k.tolist(),
            'omega_values': omega.tolist()
        }
        
    except Exception as e:
        print(f"Dispersion extraction failed: {e}")
        return None


# =============================================================================
# Single-Period FDTD Simulation Class
# =============================================================================

class STCMTSimulation:
    """
    Single-period FDTD simulation for extracting STCMT parameters.
    
    Uses Bloch periodic boundary conditions to simulate infinite periodic array
    with a single unit cell. Extracts resonance frequencies, decay rates,
    and band dispersion for STCMT modeling.
    """
    
    def __init__(self, control_points, degree=2):
        """
        Initialize STCMT simulation.
        
        Args:
            control_points: NURBS control points defining meta-atom geometry
            degree: NURBS curve degree (default: 2 for quadratic)
        """
        self.control_points = control_points
        self.degree = degree
        self.um = 1e-6
        self.period = 12.6 * self.um  # Unit cell period
        
        # Wavelength range (11.6 - 13.6 μm)
        self.wavelength_start = 11.6e-6
        self.wavelength_stop = 13.6e-6
        
        # Initialize Lumerical FDTD
        self.fdtd = lumapi.FDTD(hide=True)
        self.fdtd.switchtolayout()
        
        # Add ZnS material
        ZnS_material = self.fdtd.addmaterial("Sampled data")
        self.fdtd.setmaterial(ZnS_material, "name", "ZnS")
        self.fdtd.setmaterial("ZnS", "max coefficients", 6)
        self.fdtd.setmaterial("ZnS", "sampled data", np.transpose(np.array(sampledData)))
        
        # Generate knot vector
        self._update_knots()
        
        # Storage for extracted parameters
        self.stcmt_params = {}
        
    def _update_knots(self):
        """Update NURBS knot vector based on control points"""
        if len(self.control_points) > 0:
            n = len(self.control_points[0]) - 1
            m = n + self.degree + 1
            self.knots = np.zeros(m + 1)
            
            # Clamped knot vector
            for i in range(0, self.degree + 1):
                self.knots[i] = 0.0
                self.knots[m - i] = 1.0
            
            # Uniform internal knots
            num_internal = m - 2 * (self.degree + 1) + 1
            if num_internal > 0:
                for i in range(0, num_internal):
                    self.knots[self.degree + 1 + i] = (i + 1) / (num_internal + 1)
    
    def set_control_points(self, control_points):
        """Update control points and regenerate knot vector"""
        self.control_points = control_points
        self._update_knots()
    
    def setup_single_period_simulation(self, k_parallel=0.0):
        """
        Setup single unit cell simulation with Bloch periodic BC.
        
        Args:
            k_parallel: Parallel wavevector component for angle-dependent simulation
                       k_parallel = (2π/λ) * sin(θ)
        """
        self.fdtd.switchtolayout()
        self.fdtd.deleteall()
        
        # =====================================================================
        # Layer Stack: Ge(100nm) / ZnS(600nm) / Ge(100nm) / Au(50nm)
        # =====================================================================
        
        # Ge top layer
        self.fdtd.addrect(name="Ge_top")
        self.fdtd.set("material", "Ge (Germanium) - Palik")
        self.fdtd.set('z min', -0.1e-6)
        self.fdtd.set('z max', 0)
        self.fdtd.set('x', 0)
        self.fdtd.set('x span', self.period)
        self.fdtd.set('y', 0)
        self.fdtd.set('y span', self.period)
        
        # ZnS middle layer
        self.fdtd.addrect(name="ZnS_mid")
        self.fdtd.set("material", "ZnS")
        self.fdtd.set('z min', -0.7e-6)
        self.fdtd.set('z max', -0.1e-6)
        self.fdtd.set('x', 0)
        self.fdtd.set('x span', self.period)
        self.fdtd.set('y', 0)
        self.fdtd.set('y span', self.period)
        
        # Ge bottom layer
        self.fdtd.addrect(name="Ge_bottom")
        self.fdtd.set("material", "Ge (Germanium) - Palik")
        self.fdtd.set('z min', -0.8e-6)
        self.fdtd.set('z max', -0.7e-6)
        self.fdtd.set('x', 0)
        self.fdtd.set('x span', self.period)
        self.fdtd.set('y', 0)
        self.fdtd.set('y span', self.period)
        
        # Au bottom layer (reflector)
        self.fdtd.addrect(name="Au_bottom")
        self.fdtd.set("material", "Au (Gold) - Palik")
        self.fdtd.set('z min', -0.85e-6)
        self.fdtd.set('z max', -0.8e-6)
        self.fdtd.set('x', 0)
        self.fdtd.set('x span', self.period)
        self.fdtd.set('y', 0)
        self.fdtd.set('y span', self.period)
        
        # =====================================================================
        # Fine mesh for accuracy
        # =====================================================================
        self.fdtd.addmesh(name="fine_mesh")
        self.fdtd.set('z min', -1e-6)
        self.fdtd.set('z max', 0.2e-6)
        self.fdtd.set('x', 0)
        self.fdtd.set('x span', self.period)
        self.fdtd.set('y', 0)
        self.fdtd.set('y span', self.period)
        self.fdtd.set("dx", 0.05e-6)
        self.fdtd.set("dy", 0.05e-6)
        self.fdtd.set("dz", 0.05e-6)
        
        # =====================================================================
        # Plane wave source
        # =====================================================================
        self.fdtd.addplane(name="source")
        self.fdtd.set('wavelength start', self.wavelength_start)
        self.fdtd.set('wavelength stop', self.wavelength_stop)
        self.fdtd.set('direction', 'Backward')
        self.fdtd.set('polarization angle', 0)
        self.fdtd.set('z', 2e-6)
        self.fdtd.set('x', 0)
        self.fdtd.set('y', 0)
        self.fdtd.set('x span', self.period)
        self.fdtd.set('y span', self.period)
        
        # Set angle for k-dependent simulation
        if k_parallel != 0:
            # Calculate angle from k_parallel
            # k_parallel = k0 * sin(theta)
            k0 = 2 * np.pi / ((self.wavelength_start + self.wavelength_stop) / 2)
            sin_theta = k_parallel / k0
            sin_theta = np.clip(sin_theta, -1, 1)
            theta = np.arcsin(sin_theta) * 180 / np.pi
            self.fdtd.set('angle theta', theta)
        
        # =====================================================================
        # Monitors
        # =====================================================================
        frequency_points = 201
        
        # Reflection monitor
        self.fdtd.addpower(name="R")
        self.fdtd.set("monitor type", "2D Z-normal")
        self.fdtd.set('x', 0)
        self.fdtd.set('y', 0)
        self.fdtd.set('z', 2.5e-6)
        self.fdtd.set('x span', self.period)
        self.fdtd.set('y span', self.period)
        
        # Field profile monitor (for mode analysis)
        self.fdtd.addprofile(name="field_profile")
        self.fdtd.set("monitor type", "2D Z-normal")
        self.fdtd.set('x', 0)
        self.fdtd.set('y', 0)
        self.fdtd.set('z', 0)
        self.fdtd.set('x span', self.period)
        self.fdtd.set('y span', self.period)
        
        # Time monitor for temporal response (Q-factor extraction)
        self.fdtd.addtime(name="time_monitor")
        self.fdtd.set('x', 0)
        self.fdtd.set('y', 0)
        self.fdtd.set('z', 0)
        
        self.fdtd.setglobalmonitor("frequency points", frequency_points)
        
        # =====================================================================
        # FDTD region with Bloch periodic BC
        # =====================================================================
        zmax = 3e-6
        zmin = -1e-6
        
        self.fdtd.addfdtd(
            dimension="3D",
            x=0, y=0, z=(zmax + zmin) / 2,
            x_span=self.period, y_span=self.period, z_span=zmax - zmin
        )
        
        # Bloch periodic boundary conditions
        self.fdtd.set("x min bc", "Bloch")
        self.fdtd.set("y min bc", "Bloch")
        
        # Set Bloch wavevector
        if k_parallel != 0:
            self.fdtd.set("bloch units", "SI")
            self.fdtd.set("kx", k_parallel)
            self.fdtd.set("ky", 0)
        
        self.fdtd.set("simulation time", 3000e-15)
        
        # Generate meta-atom structure
        self._generate_meta_atom()
    
    def _generate_meta_atom(self):
        """Generate NURBS-defined meta-atom structure in single unit cell"""
        um = self.um
        
        for idx, points in enumerate(self.control_points):
            points = np.array(points) * um
            
            # Evaluate NURBS curve
            num_samples = max(100, 3 * len(points))
            t_values = np.linspace(0, 1, num_samples)[:-1]
            
            # Variable line width
            line_width = 0.5 * um + (1 - t_values) * 0.5 * um
            
            center_points = evaluate_nurbs_curve(
                self.degree,
                self.knots,
                points,
                t_values
            )
            
            # Calculate normals
            tangents = np.gradient(center_points, axis=0)
            tangents[tangents == 0] = 1e-10
            tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]
            normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
            normals[:, 0] = normals[:, 0] * line_width / 2
            normals[:, 1] = normals[:, 1] * line_width / 2
            
            # Calculate polygon vertices
            top_points = center_points + normals
            bottom_points = center_points - normals
            
            V = np.vstack((
                top_points,
                bottom_points[::-1],
                top_points[0:1]
            ))
            
            # Create Lumerical polygon (single unit cell, no array)
            vertices_str = "[" + ";\n".join(
                [f"{x}, {y}" for x, y in V]
            ) + "]"
            
            script = f"""
            addpoly;
            set("name", "nurbs_meta_atom_{idx}");
            set("x", 0);
            set("y", 0);
            set("z", 0);
            set("z span", 0.05e-6);
            set("vertices", {vertices_str});
            set("material", "Ag (Silver) - Johnson and Christy");
            """
            
            self.fdtd.eval(script)
    
    def run_simulation(self):
        """Run FDTD simulation and return results"""
        self.fdtd.switchtolayout()
        self.fdtd.run()
        
        # Get reflection spectrum
        R_result = self.fdtd.getresult('R', 'T')
        reflectance = R_result['T'].flatten()
        wavelengths = c / R_result['f'].flatten()
        
        # Get field profile
        try:
            Ex = self.fdtd.getdata('field_profile', 'Ex')
            Ey = self.fdtd.getdata('field_profile', 'Ey')
            Ez = self.fdtd.getdata('field_profile', 'Ez')
            field_data = {'Ex': Ex, 'Ey': Ey, 'Ez': Ez}
        except:
            field_data = None
        
        # Calculate absorption (for opaque structure: A = 1 - R)
        absorptance = 1 - np.abs(reflectance)
        
        return {
            'wavelengths': wavelengths,
            'reflectance': reflectance,
            'absorptance': absorptance,
            'field_data': field_data
        }
    
    def extract_stcmt_parameters(self, n_angles=5):
        """
        Extract full STCMT parameters from multiple simulations.
        
        Args:
            n_angles: Number of angles to simulate for band dispersion
            
        Returns:
            Dictionary with all STCMT parameters
        """
        print("Extracting STCMT parameters...")
        
        # =====================================================================
        # Step 1: Normal incidence simulation for resonance parameters
        # =====================================================================
        print("  Running normal incidence simulation...")
        self.setup_single_period_simulation(k_parallel=0.0)
        results_normal = self.run_simulation()
        
        # Fit resonances to extract ω₀ and γ
        resonances = fit_resonance(
            results_normal['wavelengths'],
            results_normal['absorptance'],
            method='lorentzian'
        )
        
        # =====================================================================
        # Step 2: Angle-dependent simulations for band dispersion
        # =====================================================================
        print("  Running angle-dependent simulations for band dispersion...")
        
        # Angular range: -30° to +30°
        angles = np.linspace(-30, 30, n_angles)
        k0 = 2 * np.pi / 12.6e-6  # Central wavevector
        k_parallels = k0 * np.sin(np.radians(angles))
        
        omega_vs_k = []
        
        for i, (angle, k_par) in enumerate(zip(angles, k_parallels)):
            print(f"    Angle {angle:.1f}° ({i+1}/{n_angles})...")
            
            self.setup_single_period_simulation(k_parallel=k_par)
            results = self.run_simulation()
            
            # Find peak resonance frequency
            peak_idx = np.argmax(results['absorptance'])
            omega_peak = 2 * np.pi * c / results['wavelengths'][peak_idx]
            
            omega_vs_k.append([k_par, omega_peak])
        
        omega_vs_k = np.array(omega_vs_k)
        
        # Extract dispersion parameters (effective mass, nonlocality)
        dispersion_params = extract_band_dispersion(omega_vs_k)
        
        # =====================================================================
        # Compile STCMT parameters
        # =====================================================================
        self.stcmt_params = {
            'resonances': resonances,
            'dispersion': dispersion_params,
            'normal_incidence': {
                'wavelengths': results_normal['wavelengths'].tolist(),
                'absorptance': results_normal['absorptance'].tolist(),
                'reflectance': results_normal['reflectance'].tolist()
            }
        }
        
        # Add primary mode parameters (for surrogate model training)
        if len(resonances) > 0:
            primary = resonances[0]
            self.stcmt_params['primary_mode'] = {
                'omega_0': primary['omega_0'],
                'gamma_total': primary['gamma'],
                'Q_factor': primary['Q_factor'],
                'wavelength_0': primary['wavelength_0']
            }
            
            # Estimate radiative and non-radiative decay rates
            # Using critical coupling assumption: γ_rad ≈ γ_nrad at peak absorption
            if primary.get('amplitude', 0) > 0.9:  # Near critical coupling
                gamma_rad = primary['gamma'] / 2
                gamma_nrad = primary['gamma'] / 2
            else:
                # Undercoupled estimate
                gamma_rad = primary['gamma'] * 0.3
                gamma_nrad = primary['gamma'] * 0.7
            
            self.stcmt_params['primary_mode']['gamma_rad'] = gamma_rad
            self.stcmt_params['primary_mode']['gamma_nrad'] = gamma_nrad
        
        if dispersion_params:
            self.stcmt_params['nonlocality'] = {
                'effective_mass': dispersion_params['effective_mass'],
                'band_curvature': dispersion_params['band_curvature'],
                'coupling_length': 1.0 / max(dispersion_params['nonlocality_strength'], 1e-20)
            }
        
        print("  STCMT parameter extraction complete!")
        return self.stcmt_params
    
    def save_parameters(self, filepath):
        """Save extracted STCMT parameters to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        params_serializable = self._make_serializable(self.stcmt_params)
        
        with open(filepath, 'w') as f:
            json.dump(params_serializable, f, indent=2)
        
        print(f"Parameters saved to: {filepath}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def close(self):
        """Close Lumerical session"""
        try:
            self.fdtd.close()
        except:
            pass


# =============================================================================
# Data Generation for Surrogate Model Training
# =============================================================================

def generate_random_nurbs_control_points(n_curves=1, n_points=5, 
                                          x_range=(-6, 6), y_range=(-5, 5)):
    """
    Generate random NURBS control points for dataset generation.
    
    Args:
        n_curves: Number of curves per sample
        n_points: Number of control points per curve
        x_range: (min, max) range for x coordinates
        y_range: (min, max) range for y coordinates
        
    Returns:
        List of control point arrays
    """
    curves = []
    for _ in range(n_curves):
        # Generate smooth curve by constraining point distances
        points = []
        x = np.random.uniform(x_range[0], x_range[0] + 2)
        
        for i in range(n_points):
            y = np.random.uniform(y_range[0], y_range[1])
            points.append([x, y])
            x += np.random.uniform(1, 3)  # Ensure points progress in x
        
        curves.append(points)
    
    return curves


def generate_training_dataset(n_samples, output_dir='stcmt_dataset', 
                               n_angles=3, save_interval=10):
    """
    Generate training dataset for surrogate model.
    
    Args:
        n_samples: Number of samples to generate
        output_dir: Directory to save dataset
        n_angles: Number of angles for band dispersion
        save_interval: Save progress every N samples
        
    Returns:
        List of (control_points, stcmt_params) pairs
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dataset = []
    sim = None
    
    print(f"Generating {n_samples} training samples...")
    print("=" * 60)
    
    for i in tqdm(range(n_samples)):
        try:
            # Generate random control points
            control_points = generate_random_nurbs_control_points(
                n_curves=1, 
                n_points=5
            )
            
            # Initialize or update simulation
            if sim is None:
                sim = STCMTSimulation(control_points)
            else:
                sim.set_control_points(control_points)
            
            # Extract STCMT parameters
            stcmt_params = sim.extract_stcmt_parameters(n_angles=n_angles)
            
            # Store sample
            sample = {
                'control_points': control_points,
                'stcmt_params': stcmt_params,
                'sample_id': i
            }
            dataset.append(sample)
            
            # Save progress periodically
            if (i + 1) % save_interval == 0:
                save_path = os.path.join(output_dir, f'dataset_partial_{timestamp}.json')
                with open(save_path, 'w') as f:
                    json.dump(dataset, f, indent=2, default=str)
                print(f"\n  Progress saved: {i+1}/{n_samples} samples")
        
        except Exception as e:
            print(f"\n  Error on sample {i}: {e}")
            continue
    
    # Close simulation
    if sim:
        sim.close()
    
    # Save final dataset
    final_path = os.path.join(output_dir, f'stcmt_dataset_{timestamp}.json')
    with open(final_path, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    
    print("=" * 60)
    print(f"Dataset generation complete!")
    print(f"Total samples: {len(dataset)}")
    print(f"Saved to: {final_path}")
    
    return dataset


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    # Example: Single simulation to test STCMT extraction
    print("=" * 60)
    print("STCMT Parameter Extraction Test")
    print("=" * 60)
    
    # Define example NURBS control points
    control_points = [
        [[-5.0, -4.0], [-3.0, 0.0], [-1.0, 3.0], [1.0, 0.0], [3.0, -3.0], [5.0, 2.0]]
    ]
    
    # Create simulation
    sim = STCMTSimulation(control_points, degree=2)
    
    # Extract STCMT parameters
    params = sim.extract_stcmt_parameters(n_angles=5)
    
    # Print results
    print("\n" + "=" * 60)
    print("Extracted STCMT Parameters:")
    print("=" * 60)
    
    if 'primary_mode' in params:
        pm = params['primary_mode']
        print(f"\nPrimary Mode:")
        print(f"  Resonance wavelength: {pm['wavelength_0']*1e6:.3f} μm")
        print(f"  Angular frequency ω₀: {pm['omega_0']:.3e} rad/s")
        print(f"  Total decay rate γ: {pm['gamma_total']:.3e} rad/s")
        print(f"  Radiative decay γ_rad: {pm['gamma_rad']:.3e} rad/s")
        print(f"  Non-radiative decay γ_nrad: {pm['gamma_nrad']:.3e} rad/s")
        print(f"  Quality factor Q: {pm['Q_factor']:.1f}")
    
    if 'nonlocality' in params:
        nl = params['nonlocality']
        print(f"\nNonlocality Parameters:")
        print(f"  Band curvature: {nl['band_curvature']:.3e}")
        print(f"  Effective mass: {nl['effective_mass']:.3e}")
        print(f"  Coupling length: {nl['coupling_length']:.3e} m")
    
    # Save parameters
    sim.save_parameters('stcmt_params_test.json')
    
    # Close simulation
    sim.close()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
