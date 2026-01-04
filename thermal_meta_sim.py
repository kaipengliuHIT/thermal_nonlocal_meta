"""
热辐射非局域超表面仿真 - 并行 MEEP 实现
基于 Lumerical FDTD 脚本转换

结构:
    - Ge 顶层 (100 nm)
    - ZnS 中间层 (600 nm)  
    - Ge 底层 (100 nm)
    - Au 反射层 (50 nm)
    - Ag 纳米线阵列 (NURBS B样条曲线形状)

仿真参数:
    - 波长范围: 11.6-13.6 μm
    - 周期: 12.6 μm
    - 3D FDTD + 周期边界条件

运行方式:
    单进程: python thermal_meta_sim.py
    并行:   mpirun -np 4 python thermal_meta_sim.py
"""

import numpy as np
import meep as mp
from meep import mpb
import argparse
import os

from materials_realistic import (
    create_zns_material_from_data,
    create_ge_material_palik,
    create_au_material_stable,
    create_ag_material_stable,
)
from nurbs_geometry import create_single_period_nurbs


class ThermalMetaSim:
    """热辐射超表面仿真类"""
    
    def __init__(self, control_points, line_widths=None, resolution=10):
        """
        初始化仿真
        
        Args:
            control_points: NURBS 控制点列表
            line_widths: 每条纳米线的线宽 (um)
            resolution: 网格分辨率 (pixels/um)
        """
        self.control_points = control_points
        self.resolution = resolution
        
        # 默认线宽
        if line_widths is None:
            self.line_widths = [1.1, 1.0, 0.8, 0.7, 0.6, 0.5]
        else:
            self.line_widths = line_widths
        
        # 结构参数 (um)
        self.period_x = 12.6  # x 方向周期
        self.period_y = 12.6  # y 方向周期
        
        # 层厚度 (um)
        self.ge_top_thick = 0.1
        self.zns_thick = 0.6
        self.ge_bottom_thick = 0.1
        self.au_thick = 0.05
        self.ag_height = 0.05  # 纳米线高度
        
        # 计算层位置
        self.ge_top_center = -self.ge_top_thick / 2
        self.zns_center = -self.ge_top_thick - self.zns_thick / 2
        self.ge_bottom_center = -self.ge_top_thick - self.zns_thick - self.ge_bottom_thick / 2
        self.au_center = -self.ge_top_thick - self.zns_thick - self.ge_bottom_thick - self.au_thick / 2
        
        # 仿真区域
        self.pml_thickness = 1.0  # PML 厚度
        self.z_padding = 2.0  # z 方向额外空间
        
        # 波长范围
        self.wavelength_min = 11.6  # um
        self.wavelength_max = 13.6  # um
        self.wavelength_center = (self.wavelength_min + self.wavelength_max) / 2
        
        # 频率 (MEEP 单位: c = 1)
        self.fcen = 1 / self.wavelength_center
        self.df = 1/self.wavelength_min - 1/self.wavelength_max
        self.nfreq = 201  # 频率点数
        
        # 材料 - 使用基于 Palik 数据库的真实材料参数
        self.ge_material = create_ge_material_palik()
        self.zns_material = create_zns_material_from_data()
        self.au_material = create_au_material_stable(resolution)
        self.ag_material = create_ag_material_stable(resolution)
        
        # 仿真对象
        self.sim = None
        self.refl_flux = None
        self.trans_flux = None
    
    def build_geometry(self):
        """构建仿真几何结构"""
        geometry = []
        
        # 1. Ge 顶层
        geometry.append(mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, self.ge_top_thick),
            center=mp.Vector3(0, 0, self.ge_top_center),
            material=self.ge_material
        ))
        
        # 2. ZnS 中间层
        geometry.append(mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, self.zns_thick),
            center=mp.Vector3(0, 0, self.zns_center),
            material=self.zns_material
        ))
        
        # 3. Ge 底层
        geometry.append(mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, self.ge_bottom_thick),
            center=mp.Vector3(0, 0, self.ge_bottom_center),
            material=self.ge_material
        ))
        
        # 4. Au 反射层
        geometry.append(mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, self.au_thick),
            center=mp.Vector3(0, 0, self.au_center),
            material=self.au_material
        ))
        
        # 5. Ag 纳米线阵列 (NURBS 形状)
        ag_nanowires = create_single_period_nurbs(
            control_points_list=self.control_points,
            line_widths=self.line_widths[:len(self.control_points)],
            z_center=self.ag_height / 2,
            z_height=self.ag_height,
            material=self.ag_material
        )
        geometry.extend(ag_nanowires)
        
        return geometry
    
    def build_cell(self):
        """构建仿真单元"""
        # 计算 z 方向尺寸
        total_layer_thick = (self.ge_top_thick + self.zns_thick + 
                           self.ge_bottom_thick + self.au_thick)
        z_size = total_layer_thick + 2 * self.z_padding + 2 * self.pml_thickness
        
        cell_size = mp.Vector3(self.period_x, self.period_y, z_size)
        
        return cell_size
    
    def build_sources(self):
        """构建光源"""
        # 平面波光源，从上方入射
        source_z = self.z_padding / 2 + self.pml_thickness / 2
        
        sources = [
            mp.Source(
                src=mp.GaussianSource(self.fcen, fwidth=self.df),
                component=mp.Ex,
                center=mp.Vector3(0, 0, source_z),
                size=mp.Vector3(self.period_x, self.period_y, 0)
            )
        ]
        
        return sources, source_z
    
    def setup_simulation(self):
        """设置仿真"""
        cell_size = self.build_cell()
        geometry = self.build_geometry()
        sources, source_z = self.build_sources()
        
        # PML 边界层 (只在 z 方向)
        pml_layers = [mp.PML(thickness=self.pml_thickness, direction=mp.Z)]
        
        # 创建仿真对象
        self.sim = mp.Simulation(
            cell_size=cell_size,
            geometry=geometry,
            sources=sources,
            resolution=self.resolution,
            boundary_layers=pml_layers,
            k_point=mp.Vector3(0, 0, 0),  # 正入射
            symmetries=[]  # 可以添加对称性加速
        )
        
        # 添加反射率监测器
        refl_z = source_z + 0.5
        self.refl_flux = self.sim.add_flux(
            self.fcen, self.df, self.nfreq,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, refl_z),
                size=mp.Vector3(self.period_x, self.period_y, 0)
            )
        )
        
        # 添加透射率监测器 (在金属层下方，应该为 0)
        trans_z = self.au_center - self.au_thick / 2 - 0.2
        self.trans_flux = self.sim.add_flux(
            self.fcen, self.df, self.nfreq,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, trans_z),
                size=mp.Vector3(self.period_x, self.period_y, 0)
            )
        )
        
        return source_z
    
    def run_normalization(self, until=200):
        """运行归一化仿真（无结构）"""
        if mp.am_master():
            print("Running normalization simulation (no structure)...")
        
        # 保存当前几何
        current_geometry = self.sim.geometry
        
        # 清空几何，只保留入射波
        self.sim.reset_meep()
        self.sim.geometry = []
        
        # 重新设置源和监测器
        cell_size = self.build_cell()
        sources, source_z = self.build_sources()
        
        self.sim = mp.Simulation(
            cell_size=cell_size,
            geometry=[],
            sources=sources,
            resolution=self.resolution,
            boundary_layers=[mp.PML(thickness=self.pml_thickness, direction=mp.Z)],
            k_point=mp.Vector3(0, 0, 0)
        )
        
        refl_z = source_z + 0.5
        norm_refl = self.sim.add_flux(
            self.fcen, self.df, self.nfreq,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, refl_z),
                size=mp.Vector3(self.period_x, self.period_y, 0)
            )
        )
        
        # 运行仿真
        self.sim.run(until=until)
        
        # 获取归一化数据
        norm_flux_data = self.sim.get_flux_data(norm_refl)
        incident_flux = mp.get_fluxes(norm_refl)
        
        # 恢复几何
        self.sim.reset_meep()
        self.sim.geometry = current_geometry
        
        return norm_flux_data, incident_flux
    
    def run_simulation(self, until=500):
        """运行完整仿真"""
        if mp.am_master():
            print(f"Running thermal metasurface simulation...")
            print(f"  Resolution: {self.resolution} pixels/um")
            print(f"  Wavelength range: {self.wavelength_min}-{self.wavelength_max} um")
            print(f"  Period: {self.period_x} x {self.period_y} um")
        
        # 先运行归一化
        norm_flux_data, incident_flux = self.run_normalization(until=until//2)
        
        # 重新设置仿真
        source_z = self.setup_simulation()
        
        # 加载归一化数据（用于计算反射率）
        self.sim.load_minus_flux_data(self.refl_flux, norm_flux_data)
        
        if mp.am_master():
            print("Running main simulation...")
        
        # 运行主仿真
        self.sim.run(until=until)
        
        # 获取结果
        refl_flux = mp.get_fluxes(self.refl_flux)
        trans_flux = mp.get_fluxes(self.trans_flux)
        freqs = mp.get_flux_freqs(self.refl_flux)
        
        # 计算反射率
        reflectance = np.array([-rf / inf for rf, inf in zip(refl_flux, incident_flux)])
        transmittance = np.array([tf / inf for tf, inf in zip(trans_flux, incident_flux)])
        
        # 转换为波长
        wavelengths = 1 / np.array(freqs)
        
        return wavelengths, reflectance, transmittance
    
    def run_simple(self, until=500):
        """简化的仿真运行（不做归一化）"""
        if mp.am_master():
            print(f"Running thermal metasurface simulation (simple mode)...")
        
        source_z = self.setup_simulation()
        
        if mp.am_master():
            print("Running simulation...")
        
        self.sim.run(until=until)
        
        # 获取频谱数据
        refl_flux = mp.get_fluxes(self.refl_flux)
        freqs = mp.get_flux_freqs(self.refl_flux)
        wavelengths = 1 / np.array(freqs)
        
        return wavelengths, np.array(refl_flux)
    
    def save_results(self, wavelengths, reflectance, transmittance=None, filename="results"):
        """保存结果到文件"""
        if mp.am_master():
            data = np.column_stack([wavelengths, reflectance])
            if transmittance is not None:
                data = np.column_stack([data, transmittance])
                header = "wavelength(um), reflectance, transmittance"
            else:
                header = "wavelength(um), reflectance"
            
            np.savetxt(f"{filename}.csv", data, delimiter=",", header=header)
            print(f"Results saved to {filename}.csv")
    
    def plot_results(self, wavelengths, reflectance, transmittance=None, filename="results"):
        """绘制结果（仅主进程）"""
        if mp.am_master():
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, reflectance, 'b-', label='Reflectance', linewidth=2)
            
            if transmittance is not None:
                plt.plot(wavelengths, transmittance, 'r--', label='Transmittance', linewidth=2)
                absorbance = 1 - reflectance - transmittance
                plt.plot(wavelengths, absorbance, 'g-.', label='Absorbance', linewidth=2)
            
            plt.xlabel('Wavelength (μm)', fontsize=12)
            plt.ylabel('Intensity', fontsize=12)
            plt.title('Thermal Metasurface Spectrum', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim([self.wavelength_min, self.wavelength_max])
            plt.ylim([0, 1])
            
            plt.savefig(f"{filename}.png", dpi=150, bbox_inches='tight')
            print(f"Plot saved to {filename}.png")
            plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Thermal Metasurface Simulation with MEEP')
    parser.add_argument('--resolution', type=int, default=10, 
                        help='Grid resolution (pixels/um)')
    parser.add_argument('--until', type=float, default=500,
                        help='Simulation time')
    parser.add_argument('--output', type=str, default='results',
                        help='Output filename prefix')
    parser.add_argument('--simple', action='store_true',
                        help='Run simple simulation without normalization')
    args = parser.parse_args()
    
    # 定义 NURBS 控制点 (与原 Lumerical 脚本相同)
    d = np.array([0.2, 0.3, 0.5, 0.7, 1.0, 1.0])
    l = np.array([5.0, 4.8, 4.6, 3.5, 3.0, 3.0])
    
    control_points = [
        [[-5.25 + d[0], -l[0]], [-5.25, 0], [-5.25 + d[0], l[0]]],
        [[-3.15 + d[1], -l[1]], [-3.15, 0], [-3.15 + d[1], l[1]]],
        [[-1.05 + d[2], -l[2]], [-1.05, 0], [-1.05 + d[2], l[2]]],
        [[1.05 + d[3], -l[3]], [1.05, 0], [1.05 + d[3], l[3]]],
        [[3.15 + d[4], -l[4]], [3.15, 0], [3.15 + d[4], l[4]]],
        [[5.25 + d[5], -l[5]], [5.25, 0], [5.25 + d[5], l[5]]]
    ]
    
    line_widths = [1.1, 1.0, 0.8, 0.7, 0.6, 0.5]
    
    # 创建仿真
    sim = ThermalMetaSim(
        control_points=control_points,
        line_widths=line_widths,
        resolution=args.resolution
    )
    
    # 运行仿真
    if args.simple:
        wavelengths, flux = sim.run_simple(until=args.until)
        sim.save_results(wavelengths, flux, filename=args.output)
    else:
        wavelengths, reflectance, transmittance = sim.run_simulation(until=args.until)
        sim.save_results(wavelengths, reflectance, transmittance, filename=args.output)
        sim.plot_results(wavelengths, reflectance, transmittance, filename=args.output)
    
    if mp.am_master():
        print("Simulation completed!")


if __name__ == "__main__":
    main()
