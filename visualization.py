"""
可视化模块 - 几何结构、电场分布、吸收谱可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import meep as mp
import h5py
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


def plot_geometry_2d(sim, output_file="geometry_2d.png", z_plane=0, 
                     figsize=(12, 10), dpi=150):
    """
    绘制仿真几何结构的 2D 截面图
    
    Args:
        sim: MEEP Simulation 对象
        output_file: 输出文件名
        z_plane: z 截面位置
        figsize: 图像尺寸
        dpi: 分辨率
    """
    if not mp.am_master():
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用 MEEP 的绘图功能
    sim.plot2D(ax=ax, output_plane=mp.Volume(center=mp.Vector3(0, 0, z_plane),
                                              size=mp.Vector3(sim.cell_size.x, 
                                                              sim.cell_size.y, 0)))
    
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_title(f'Geometry Cross-section at z = {z_plane} μm', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Geometry plot saved to {output_file}")


def plot_geometry_3d_schematic(control_points, line_widths, layer_structure,
                               output_file="geometry_3d.png", figsize=(14, 10)):
    """
    绘制结构的 3D 示意图
    
    Args:
        control_points: NURBS 控制点列表
        line_widths: 线宽列表
        layer_structure: 层结构字典 {name: (z_min, z_max, color)}
        output_file: 输出文件名
    """
    if not mp.am_master():
        return
    
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制层结构
    period = 12.6
    x_range = [-period/2, period/2]
    y_range = [-period/2, period/2]
    
    for name, (z_min, z_max, color) in layer_structure.items():
        # 创建长方体的 6 个面
        vertices = [
            # 底面
            [[x_range[0], y_range[0], z_min],
             [x_range[1], y_range[0], z_min],
             [x_range[1], y_range[1], z_min],
             [x_range[0], y_range[1], z_min]],
            # 顶面
            [[x_range[0], y_range[0], z_max],
             [x_range[1], y_range[0], z_max],
             [x_range[1], y_range[1], z_max],
             [x_range[0], y_range[1], z_max]],
        ]
        
        for v in vertices:
            ax.add_collection3d(Poly3DCollection([v], alpha=0.3, 
                                                  facecolor=color, 
                                                  edgecolor='gray',
                                                  linewidth=0.5))
    
    # 绘制 NURBS 纳米线
    from nurbs_geometry import generate_nurbs_curve, curve_to_polygon_vertices
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(control_points)))
    
    for idx, (points, width) in enumerate(zip(control_points, line_widths)):
        center_points = generate_nurbs_curve(points, num_points_per_segment=30)
        if center_points is not None:
            # 绘制曲线
            ax.plot(center_points[:, 0], center_points[:, 1], 
                   np.ones(len(center_points)) * 0.025,
                   color=colors[idx], linewidth=width*2, label=f'Wire {idx+1}')
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_zlabel('Z (μm)')
    ax.set_title('3D Structure Schematic')
    
    # 添加图例
    legend_elements = [mpatches.Patch(facecolor=color, alpha=0.3, 
                                       edgecolor='gray', label=name)
                       for name, (_, _, color) in layer_structure.items()]
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim([-1, 0.5])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"3D schematic saved to {output_file}")


def plot_structure_layers(output_file="structure_layers.png"):
    """
    绘制结构层示意图（侧视图）
    """
    if not mp.am_master():
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 层结构参数
    layers = [
        ('Ag nanowires', 0, 0.05, 'silver', 'NURBS curves'),
        ('Ge top', -0.1, 0, 'darkgray', 'n ≈ 4.0'),
        ('ZnS', -0.7, -0.1, 'lightgreen', 'n ≈ 2.2'),
        ('Ge bottom', -0.8, -0.7, 'darkgray', 'n ≈ 4.0'),
        ('Au mirror', -0.85, -0.8, 'gold', 'Reflector'),
    ]
    
    x_width = 6
    
    for name, z_min, z_max, color, note in layers:
        rect = plt.Rectangle((-x_width/2, z_min), x_width, z_max - z_min,
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        
        # 添加标签
        z_center = (z_min + z_max) / 2
        ax.text(x_width/2 + 0.3, z_center, f'{name}\n({(z_max-z_min)*1000:.0f} nm)',
               fontsize=10, va='center')
        ax.text(-x_width/2 - 0.3, z_center, note,
               fontsize=9, va='center', ha='right', style='italic')
    
    # 添加入射光箭头
    ax.annotate('', xy=(0, 0.3), xytext=(0, 0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.3, 0.55, 'Incident\nlight', fontsize=10, color='red')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.2, 1)
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Z (μm)', fontsize=12)
    ax.set_title('Thermal Metasurface Structure (Side View)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Structure layers plot saved to {output_file}")


def plot_nurbs_curves(control_points, line_widths, output_file="nurbs_curves.png"):
    """
    绘制 NURBS 曲线形状
    """
    if not mp.am_master():
        return
    
    from nurbs_geometry import generate_nurbs_curve, curve_to_polygon_vertices
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: 曲线和控制点
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(control_points)))
    
    for idx, (points, width) in enumerate(zip(control_points, line_widths)):
        points_array = np.array(points)
        
        # 绘制控制点
        ax1.scatter(points_array[:, 0], points_array[:, 1], 
                   color=colors[idx], s=100, zorder=5, marker='o')
        ax1.plot(points_array[:, 0], points_array[:, 1], 
                '--', color=colors[idx], alpha=0.5, linewidth=1)
        
        # 绘制曲线
        center_points = generate_nurbs_curve(points, num_points_per_segment=50)
        if center_points is not None:
            ax1.plot(center_points[:, 0], center_points[:, 1], 
                    color=colors[idx], linewidth=2, 
                    label=f'Curve {idx+1} (w={width}μm)')
    
    ax1.set_xlabel('X (μm)', fontsize=12)
    ax1.set_ylabel('Y (μm)', fontsize=12)
    ax1.set_title('NURBS Curves and Control Points', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 右图: 带线宽的多边形
    ax2 = axes[1]
    
    for idx, (points, width) in enumerate(zip(control_points, line_widths)):
        center_points = generate_nurbs_curve(points, num_points_per_segment=50)
        if center_points is not None:
            vertices = curve_to_polygon_vertices(center_points, width)
            
            # 绘制多边形
            polygon = plt.Polygon(vertices, facecolor=colors[idx], 
                                  edgecolor='black', alpha=0.7, linewidth=0.5)
            ax2.add_patch(polygon)
    
    ax2.set_xlabel('X (μm)', fontsize=12)
    ax2.set_ylabel('Y (μm)', fontsize=12)
    ax2.set_title('Nanowire Polygons with Line Width', fontsize=14)
    ax2.set_xlim(-7, 7)
    ax2.set_ylim(-6, 6)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"NURBS curves plot saved to {output_file}")


def plot_field_distribution(sim, component=mp.Ez, z_plane=0, 
                           output_file="field_Ez.png", figsize=(12, 10)):
    """
    绘制电场分布
    
    Args:
        sim: MEEP Simulation 对象
        component: 场分量 (mp.Ex, mp.Ey, mp.Ez, etc.)
        z_plane: z 截面位置
        output_file: 输出文件名
    """
    if not mp.am_master():
        return
    
    # 获取场数据
    field_data = sim.get_array(center=mp.Vector3(0, 0, z_plane),
                               size=mp.Vector3(sim.cell_size.x, sim.cell_size.y, 0),
                               component=component)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    extent = [-sim.cell_size.x/2, sim.cell_size.x/2,
              -sim.cell_size.y/2, sim.cell_size.y/2]
    
    # 实部
    ax1 = axes[0]
    im1 = ax1.imshow(np.real(field_data).T, extent=extent, origin='lower',
                     cmap='RdBu_r', aspect='equal')
    plt.colorbar(im1, ax=ax1, label='Real part')
    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_title(f'Re({component.name}) at z = {z_plane} μm')
    
    # 幅度
    ax2 = axes[1]
    im2 = ax2.imshow(np.abs(field_data).T, extent=extent, origin='lower',
                     cmap='hot', aspect='equal')
    plt.colorbar(im2, ax=ax2, label='|E|')
    ax2.set_xlabel('X (μm)')
    ax2.set_ylabel('Y (μm)')
    ax2.set_title(f'|{component.name}| at z = {z_plane} μm')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Field distribution plot saved to {output_file}")


def plot_epsilon_distribution(sim, z_plane=0, output_file="epsilon.png"):
    """
    绘制介电常数分布
    """
    if not mp.am_master():
        return
    
    # 获取介电常数分布
    eps_data = sim.get_array(center=mp.Vector3(0, 0, z_plane),
                             size=mp.Vector3(sim.cell_size.x, sim.cell_size.y, 0),
                             component=mp.Dielectric)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    extent = [-sim.cell_size.x/2, sim.cell_size.x/2,
              -sim.cell_size.y/2, sim.cell_size.y/2]
    
    im = ax.imshow(np.real(eps_data).T, extent=extent, origin='lower',
                   cmap='viridis', aspect='equal')
    plt.colorbar(im, ax=ax, label='ε (dielectric constant)')
    
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_title(f'Dielectric Distribution at z = {z_plane} μm', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Epsilon distribution plot saved to {output_file}")


def plot_spectrum(wavelengths, reflectance, transmittance=None, 
                  output_file="spectrum.png", figsize=(12, 8)):
    """
    绘制光谱 (反射率、透射率、吸收率)
    
    Args:
        wavelengths: 波长数组 (μm)
        reflectance: 反射率数组
        transmittance: 透射率数组 (可选)
        output_file: 输出文件名
    """
    if not mp.am_master():
        return
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # 上图: R, T, A
    ax1 = axes[0]
    ax1.plot(wavelengths, reflectance, 'b-', linewidth=2, label='Reflectance (R)')
    
    if transmittance is not None:
        ax1.plot(wavelengths, transmittance, 'g--', linewidth=2, label='Transmittance (T)')
        absorbance = 1 - reflectance - transmittance
        absorbance = np.clip(absorbance, 0, 1)  # 确保在 [0,1] 范围内
        ax1.plot(wavelengths, absorbance, 'r-.', linewidth=2, label='Absorbance (A)')
    
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title('Thermal Metasurface Optical Spectrum', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # 下图: 发射率 (= 吸收率，根据基尔霍夫定律)
    ax2 = axes[1]
    if transmittance is not None:
        emissivity = 1 - reflectance - transmittance
        emissivity = np.clip(emissivity, 0, 1)
    else:
        emissivity = 1 - reflectance
        emissivity = np.clip(emissivity, 0, 1)
    
    ax2.fill_between(wavelengths, emissivity, alpha=0.3, color='red')
    ax2.plot(wavelengths, emissivity, 'r-', linewidth=2, label='Emissivity (ε)')
    
    ax2.set_xlabel('Wavelength (μm)', fontsize=12)
    ax2.set_ylabel('Emissivity', fontsize=12)
    ax2.set_title('Thermal Emissivity Spectrum (Kirchhoff\'s Law: ε = A)', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Spectrum plot saved to {output_file}")


def plot_spectrum_simple(csv_file, output_file="spectrum_simple.png"):
    """
    从 CSV 文件绘制简单光谱图
    """
    if not mp.am_master():
        return
    
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    wavelengths = data[:, 0]
    flux = data[:, 1]
    
    # 归一化
    flux_norm = flux / np.max(np.abs(flux))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(wavelengths, flux_norm, 'b-', linewidth=2)
    ax.fill_between(wavelengths, flux_norm, alpha=0.3)
    
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Normalized Flux', fontsize=12)
    ax.set_title('Reflected Flux Spectrum', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Simple spectrum plot saved to {output_file}")


def generate_all_visualizations(control_points, line_widths, output_dir="."):
    """
    生成所有静态可视化图
    
    Args:
        control_points: NURBS 控制点
        line_widths: 线宽列表
        output_dir: 输出目录
    """
    if not mp.am_master():
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    print("=" * 50)
    
    # 1. 结构层示意图
    plot_structure_layers(os.path.join(output_dir, "structure_layers.png"))
    
    # 2. NURBS 曲线图
    plot_nurbs_curves(control_points, line_widths,
                      os.path.join(output_dir, "nurbs_curves.png"))
    
    # 3. 3D 示意图
    layer_structure = {
        'Ag wires': (0, 0.05, 'silver'),
        'Ge top': (-0.1, 0, 'gray'),
        'ZnS': (-0.7, -0.1, 'lightgreen'),
        'Ge bottom': (-0.8, -0.7, 'gray'),
        'Au mirror': (-0.85, -0.8, 'gold'),
    }
    plot_geometry_3d_schematic(control_points, line_widths, layer_structure,
                               os.path.join(output_dir, "geometry_3d.png"))
    
    print("=" * 50)
    print("All static visualizations generated!")


if __name__ == "__main__":
    import numpy as np
    
    # 测试控制点
    d = np.array([0.2, 0.3, 0.5, 0.7, 1.0, 1.0])
    l = np.array([5.0, 4.8, 4.6, 3.5, 3.0, 3.0])
    
    control_points = [
        [[-5.25 + d[i], -l[i]], [-5.25 + 2.1*i, 0], [-5.25 + d[i], l[i]]]
        for i in range(6)
    ]
    line_widths = [1.1, 1.0, 0.8, 0.7, 0.6, 0.5]
    
    # 生成可视化
    generate_all_visualizations(control_points, line_widths, "figures")
