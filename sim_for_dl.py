import numpy as np

# Load data from txt file
data = np.loadtxt('ZnS.txt')  # Ensure file is in current working directory or provide full path

# Extract columns
wavelength_um = data[:, 0]  # Column 1: wavelength (micrometers)
n_real = data[:, 1]         # Column 2: real part of refractive index
n_imag = data[:, 2]        # Column 3: imaginary part of refractive index

# Calculate complex refractive index (n + ik)
refractive_index_complex = n_real + 1j * n_imag  # Using Python built-in complex representation

# Calculate complex relative permittivity (ε = ñ² = (n + ik)² = n² - k² + i2nk)
epsilon_complex = refractive_index_complex**2

# Convert wavelength to frequency
# c = λf => f = c/λ
c = 3e8  # Speed of light (m/s)
wavelength_m = wavelength_um * 1e-6  # Convert micrometers to meters
f = c / wavelength_m  # Calculate frequency (Hz)

# Create final data structure
sampledData = [f, epsilon_complex]

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib
import math
import random
from scipy.spatial import ConvexHull

#lumapi = importlib.machinery.SourceFileLoader('lumapi', 'E:\\Program Files\\Lumerical\\v241\\api\\python\\lumapi.py').load_module()
lumapi = importlib.machinery.SourceFileLoader('lumapi', 'C:\\Program Files\\Lumerical\\v241\\api\\python\\lumapi.py').load_module()
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

def basis_function_vectorized(i, p, knots, t):
    """Vectorized B-spline basis function calculation (iterative implementation)"""
    N = np.zeros(len(t))
    # 0-th order basis function
    N[(knots[i] <= t) & (t < knots[i+1])] = 1.0
    
    # Recursively calculate higher order basis functions
    for k in range(1, p+1):
        # Temporary storage for current order basis function values
        N_temp = np.zeros(len(t))
        
        # Calculate current order basis function
        for j in range(i, i+p+1):
            denom1 = knots[j+k] - knots[j]
            denom2 = knots[j+k+1] - knots[j+1]
            
            term1 = 0.0
            term2 = 0.0
            
            if denom1 != 0:
                term1 = (t - knots[j]) / denom1 * N[j-i]
            
            if denom2 != 0:
                term2 = (knots[j+k+1] - t) / denom2 * N[j-i+1]
            
            N_temp[j-i] = term1 + term2
        
        # Update basis function values
        N = N_temp
    
    return N

def find_span(n, p, knots, t):
    """Find the knot span containing parameter t"""
    # Special case handling
    if t >= knots[n+1]:
        return n
    if t <= knots[p]:
        return p
    
    # Binary search
    low = p
    high = n+1
    mid = (low + high) // 2
    
    while t < knots[mid] or t >= knots[mid+1]:
        if t < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    
    return mid

def evaluate_nurbs_curve(p, knots, control_points, t_values):
    n = len(control_points) - 1
    curve_points = []
    # print(control_points)
    # print(t_values)
    for t in t_values:
        # Find the knot span containing t
        span = find_span(n, p, knots, t)
        
        # Calculate non-zero basis functions
        N = np.zeros(p+1)
        for i in range(0, p+1):
            N[i] = basis_function(span-p+i, p, knots, t)
        
        # Calculate curve point
        point = np.zeros(2)
        for i in range(0, p+1):
            idx = span - p + i
            point += N[i] * np.array(control_points[idx])
        
        curve_points.append(point)
    
    return np.array(curve_points)

class Simulation:
    def __init__(self, control_points, degree=2, target_phase=0):
        self.control_points = control_points  # (N,2) control point coordinates
        self.degree = degree  # NURBS curve degree
        self.fdtd = lumapi.FDTD(hide=False)
        self.um = 1e-6
        self.target_phase = target_phase
        self.fdtd.switchtolayout()
        self.phase = 0
        self.fdtd.save("nurbs_thermal_m.fsp")
        ZnS_material = self.fdtd.addmaterial("Sampled data")
        self.fdtd.setmaterial(ZnS_material,"name","ZnS")
        self.fdtd.setmaterial("ZnS","max coefficients",6)
        self.fdtd.setmaterial("ZnS","sampled data",np.transpose(np.array(sampledData)))
        # Generate knot vector
        n = len(control_points) - 1
        m = n + self.degree + 1
        self.knots = np.zeros(m+1)
        
        # Uniform knot vector (clamped)
        for i in range(0, self.degree+1):
            self.knots[i] = 0.0
            self.knots[m-i] = 1.0
        
        # Internal knots uniformly distributed
        num_internal = m - 2*(self.degree+1) + 1
        if num_internal > 0:
            for i in range(0, num_internal):
                self.knots[self.degree+1+i] = (i+1) / (num_internal+1)

    def set_control_points(self, control_points):
        self.control_points = control_points
        # Update knot vector
        n = len(self.control_points[0]) - 1
        m = n + self.degree + 1
        self.knots = np.zeros(m+1)
        # Uniform knot vector (clamped)
        for i in range(0, self.degree+1):
            self.knots[i] = 0.0
            self.knots[m-i] = 1.0
        
        # Internal knots uniformly distributed
        num_internal = m - 2*(self.degree+1) + 1
        if num_internal > 0:
            for i in range(0, num_internal):
                self.knots[self.degree+1+i] = (i+1) / (num_internal+1)
        # print("self.knots: ",self.knots)

    def set_target_phase(self, target_phase):
        self.target_phase = target_phase
        
    def setup_simulation(self):
        """Basic simulation setup"""
        self.fdtd.switchtolayout()
        self.fdtd.deleteall()

        # Ge top layer setup
        self.fdtd.addrect(name="Ge_top")
        self.fdtd.set("material","Ge (Germanium) - Palik")
        self.fdtd.set('z min',-0.1e-6)
        self.fdtd.set('z max',0)
        self.fdtd.set('x',0)
        self.fdtd.set('x span',70e-6)
        self.fdtd.set('y',0)
        self.fdtd.set('y span',70e-6)

        # ZnS middle layer setup
        self.fdtd.addrect(name="ZnS_mid")
        self.fdtd.set("material","ZnS")
        self.fdtd.set('z min',-0.7e-6)
        self.fdtd.set('z max',-0.1e-6)
        self.fdtd.set('x',0)
        self.fdtd.set('x span',70e-6)
        self.fdtd.set('y',0)
        self.fdtd.set('y span',70e-6)

        # Ge bottom layer setup
        self.fdtd.addrect(name="Ge_bottom")
        self.fdtd.set("material","Ge (Germanium) - Palik")
        self.fdtd.set('z min',-0.8e-6)
        self.fdtd.set('z max',-0.7e-6)
        self.fdtd.set('x',0)
        self.fdtd.set('x span',70e-6)
        self.fdtd.set('y',0)
        self.fdtd.set('y span',70e-6)

        # Au bottom layer setup (reflector)
        self.fdtd.addrect(name="Au_bottom")
        self.fdtd.set("material","Au (Gold) - Palik")
        self.fdtd.set('z min',-0.85e-6)
        self.fdtd.set('z max',-0.8e-6)
        self.fdtd.set('x',0)
        self.fdtd.set('x span',70e-6)
        self.fdtd.set('y',0)
        self.fdtd.set('y span',70e-6)

        self.fdtd.addmesh(name="fine_mesh")
        self.fdtd.set('z min',-1e-6)
        self.fdtd.set('z max',0.2e-6)
        self.fdtd.set('x',0)
        self.fdtd.set('x span',70e-6)
        self.fdtd.set('y',0)
        self.fdtd.set('y span',20e-6)
        self.fdtd.set("dx",0.2e-6)
        self.fdtd.set("dy",0.2e-6)
        self.fdtd.set("dz",0.2e-6)

        self.fdtd.addplane(name="source")
        self.fdtd.set('wavelength start',11.6e-6)
        self.fdtd.set('wavelength stop',13.6e-6)
        self.fdtd.set('direction','Backward')
        self.fdtd.set('polarization angle',0)
        self.fdtd.set('z',2e-6)
        self.fdtd.set('x',0)
        self.fdtd.set('y',0)
        self.fdtd.set('x span',70e-6)
        self.fdtd.set('y span',20e-6)
        frequency_points = int(2e-6/(1e-8)) + 2
        # Add monitor
        self.fdtd.addpower(
            name="R",
            x=0, y=0, z=3e-6,
            x_span=70e-6, y_span=20e-6, 
            monitor_type="2D Z-normal"
        )
        self.fdtd.setglobalmonitor("frequency points",frequency_points)


        zmax =  4e-6
        zmin = -1e-6

        self.fdtd.addfdtd(
            dimension="3D",
            x=0, y=0, z=(zmax+zmin)/2,
            x_span=(12.6e-6)*5, y_span=12.6e-6, z_span=zmax-zmin
        )

        self.fdtd.set("y min bc","Periodic")

        self.fdtd.set("simulation time",5000e-15)

        self.generate_structure(self.control_points)   

    def generate_structure(self, curves):
        um = 1e-6

        period = 12.6 * um
        # Store vertex matrices for all curves
        all_vertices = []
        self.set_control_points(curves)
        
        for idx, points in enumerate(self.control_points):
            points = np.array(points) * um
            
            # Calculate points on the curve
            num_samples = max(100, 3 * len(points))  # Number of sample points
            t_values = np.linspace(0, 1, num_samples)
            t_values = t_values[:-1]
            line_width = 0.5*um + (1-t_values) * 0.5 * um
            # print(t_values)
            # Calculate points on the curve
            center_points = evaluate_nurbs_curve(
                self.degree, 
                self.knots, 
                points, 
                t_values
            )
            # Calculate normal direction
            tangents = np.gradient(center_points, axis=0)
            tangents[tangents==0] = 1e-10
            tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]
            normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
            normals[:,0] =  normals[:,0]* line_width/2
            normals[:,1] =  normals[:,1]* line_width/2
            # Calculate polygon vertices
            top_points = center_points + normals
            bottom_points = center_points - normals
            
            # Create polygon vertex matrix
            V = np.vstack((
                top_points,
                bottom_points[::-1],
                top_points[0:1]  # Close the polygon
            ))
            
            all_vertices.append(V)

        # Create Lumerical script
        script = "um = 1e-6;\n"
        script += "period = 12.6 * um;\n"  # Define period
        
        array_size = 5  # 5x5 array
        
        for i, V in enumerate(all_vertices):
            # Create array for each original structure
            for row in range(array_size):
                for col in range(array_size):
                    # Calculate offset
                    x_offset = (col-2) * period
                    y_offset = (row-2) * period
                    
                    # Apply offset to all vertices
                    V_offset = V.copy()

                    V_offset[:, 0] += x_offset  # X-direction offset
                    V_offset[:, 1] += y_offset  # Y-direction offset
                    
                    # Convert to Lumerical matrix format
                    vertices_str = "[" + ";\n".join(
                        [f"{x}, {y}" for x, y in V_offset]
                    ) + "]"
                    
                    name = f"nurbs_waveguide_{i}_{row}_{col}"
                    script += f"""
                                addpoly;
                                set("name", "{name}");
                                set("x", 0);
                                set("y", 0);
                                set("z", 0);
                                set("z span", 0.05e-6);
                                set("vertices", {vertices_str});
                                set("material", "Ag (Silver) - Johnson and Christy");
                                """
        self.fdtd.eval(script)
    
    def run_forward(self, wavelength_start=8e-6,wavelength_stop=14e-6):
        """Run forward simulation"""
        self.fdtd.switchtolayout()
        self.fdtd.run()
        Reflect = (self.fdtd.getresult('R','T'))["T"]
        phase = np.angle(self.fdtd.getdata('phase','Ex'))
        self.phase = phase

        return Reflect,phase