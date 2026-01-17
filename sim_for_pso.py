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

lumapi = importlib.machinery.SourceFileLoader('lumapi', 'C:\\Program Files\\Lumerical\\v241\\api\\python\\lumapi.py').load_module()
def basis_function(i, t):
    """Quadratic B-spline basis function"""
    if i == 1:
        return (1-t)*(1-t)
    if i == 2:
        return 2*t*(1-t)
    if i == 3:
        return t*t
    return 0

class Simulation:
    def __init__(self, control_points, target_phase = 0):
        self.control_points = control_points  # (N,2) control point coordinates
        self.fdtd = lumapi.FDTD(hide=False)
        self.um = 1e-6
        self.knots = np.array([0,0,0,1,2,3,4,5,6,7,7,7])
        self.edge_indices = [[0,1,2],[2,3,4],[4,5,6],[6,7,0]]
        self.target_phase = target_phase
        self.fdtd.switchtolayout()
        ZnS_material = self.fdtd.addmaterial("Sampled data")
        self.fdtd.setmaterial(ZnS_material,"name","ZnS")
        self.fdtd.setmaterial("ZnS","max coefficients",6)
        self.fdtd.setmaterial("ZnS","sampled data",np.transpose(np.array(sampledData)))
        self.phase = 0
        self.fdtd.save("nurbs_thermal02.fsp")

    def set_contral_points(self,control_points):
        self.control_points = control_points

    def set_target_phase(self,target_phase):
        self.target_phase = target_phase

    def setup_simulation(self):
        """Basic simulation setup"""
        self.fdtd.switchtolayout()
        self.fdtd.deleteall()

        # Substrate setup - Ge top layer
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
        self.fdtd.set("dx",0.1e-6)
        self.fdtd.set("dy",0.1e-6)
        self.fdtd.set("dz",0.1e-6)

        self.fdtd.addplane(name="source")
        self.fdtd.set('wavelength start',11.6e-6)
        self.fdtd.set('wavelength stop',13.6e-6)
        self.fdtd.set('direction','Backward')
        self.fdtd.set('polarization angle',0)
        self.fdtd.set('z',3e-6)
        self.fdtd.set('x',0)
        self.fdtd.set('y',0)
        self.fdtd.set('x span',70e-6)
        self.fdtd.set('y span',20e-6)
        frequency_points = int(2e-6/(1e-8)) + 2
        # Add monitor
        self.fdtd.addpower(
            name="R",
            x=0, y=0, z=4e-6,
            x_span=70e-6, y_span=20e-6, 
            monitor_type="2D Z-normal"
        )
        self.fdtd.setglobalmonitor("frequency points",frequency_points)


        zmax =  5e-6
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
        line_width = np.array([1.1*um,1*um,0.8*um,0.7*um,0.6*um,0.5*um])
        period = 12.6 * um  # Array period
        
        # Store vertex matrices for all curves
        all_vertices = []
        self.set_contral_points(curves)
        
        for idx, points in enumerate(self.control_points):
            points = np.array(points) * um
            
            # Calculate number of curve segments
            segments = len(points) - 2
            if segments < 1:
                continue  # At least 3 points required to form a curve segment
                
            num_points_per_segment = 100
            total_points = segments * num_points_per_segment
            
            # Store center points
            center_points = np.zeros((total_points, 2))
            
            # Calculate points on the curve
            for seg in range(segments):
                for j in range(num_points_per_segment):
                    t = j / (num_points_per_segment - 1)
                    
                    # Calculate point position using basis functions
                    x = basis_function(1, t) * points[seg][0] + \
                        basis_function(2, t) * points[seg+1][0] + \
                        basis_function(3, t) * points[seg+2][0]
                        
                    y = basis_function(1, t) * points[seg][1] + \
                        basis_function(2, t) * points[seg+1][1] + \
                        basis_function(3, t) * points[seg+2][1]
                        
                    index = seg * num_points_per_segment + j
                    center_points[index] = [x, y]
            
            # Calculate normal direction
            tangents = np.zeros_like(center_points)
            tangents[1:-1] = center_points[2:] - center_points[:-2]
            tangents[0] = center_points[1] - center_points[0]
            tangents[-1] = center_points[-1] - center_points[-2]
            
            # Normalize tangent vectors
            norms = np.linalg.norm(tangents, axis=1)
            norms[norms == 0] = 1e-10  # Avoid division by zero
            tangents /= norms[:, np.newaxis]
            
            # Calculate normal vectors (rotate 90 degrees)
            normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
            
            # Calculate polygon vertices
            top_points = center_points + normals * line_width[idx] / 2
            bottom_points = center_points - normals * line_width[idx] / 2
            
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
    
    
    def run_forward(self, wavelength_start=400e-9,wavelength_stop=700e-9):
        """Run forward simulation"""
        self.fdtd.switchtolayout()
        self.fdtd.run()
        Reflect = (self.fdtd.getresult('R','T'))["T"]
        phase = np.angle(self.fdtd.getdata('phase','Ex'))
        self.phase = phase

        return Reflect,phase


# if __name__ == '__main__':
#     wavelength_start=8e-6
#     wavelength_stop=12e-6
#     d = np.array([0.2,0.3,0.5,0.7,1,1])
#     l = np.array([5,4.8,4.6,3.5,3,3])
#     points2 = np.array([[[-5.25+d[0],-l[0]],[-5.25,0],[-5.25+d[0],l[0]]],
#                         [[-3.15+d[1],-l[1]],[-3.15,0],[-3.15+d[1],l[1]]],
#                         [[-1.05+d[2],-l[2]],[-1.05,0],[-1.05+d[2],l[2]]],
#                         [[1.05+d[3],-l[3]],[1.05,0],[1.05+d[3],l[3]]],
#                         [[3.15+d[4],-l[4]],[3.15,0],[3.15+d[4],l[4]]],
#                         [[5.25+d[5],-l[5]],[5.25,0],[5.25+d[5],l[5]]]
#                         ])
#     sim = Simulation(points2)
#     sim.set_contral_points(points2)
#     sim.setup_simulation()
