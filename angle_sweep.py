"""
角度扫描仿真 - 非局域超表面
用于生成角度-波长色散图（类似能带图）

关键设置：
1. 结构重复 5 个周期以上
2. 入射角度变化方向（x方向）使用 PML 边界
3. y 方向使用周期边界条件
4. 支持大规模并行 (128+ 核)

运行方式:
    mpirun -np 128 python angle_sweep.py --angles 0 5 10 15 20 25 30
"""

import numpy as np
import meep as mp
import argparse
import os
from datetime import datetime

from materials_realistic import (
    create_zns_material_from_data,
    create_ge_material_palik,
    create_au_material_stable,
    create_ag_material_stable,
)
from nurbs_geometry import create_nurbs_prism


def create_multi_period_geometry(control_points, line_widths, num_periods=5,
                                  period_x=12.6, materials=None):
    """
    创建多周期重复的几何结构
    
    Args:
        control_points: NURBS 控制点
        line_widths: 线宽列表
        num_periods: x 方向的周期数
        period_x: x 方向周期 (um)
        materials: 材料字典
    
    Returns:
        geometry 列表
    """
    geometry = []
    
    # 计算总宽度
    total_width = num_periods * period_x
    
    # 层结构参数 (um)
    ge_top_thick = 0.1
    zns_thick = 0.6
    ge_bottom_thick = 0.1
    au_thick = 0.05
    ag_height = 0.05
    
    # 层位置
    ge_top_center = -ge_top_thick / 2
    zns_center = -ge_top_thick - zns_thick / 2
    ge_bottom_center = -ge_top_thick - zns_thick - ge_bottom_thick / 2
    au_center = -ge_top_thick - zns_thick - ge_bottom_thick - au_thick / 2
    
    # 基底层 - 覆盖所有周期
    # Ge 顶层
    geometry.append(mp.Block(
        center=mp.Vector3(0, 0, ge_top_center),
        size=mp.Vector3(total_width, mp.inf, ge_top_thick),
        material=materials['ge']
    ))
    
    # ZnS 中间层
    geometry.append(mp.Block(
        center=mp.Vector3(0, 0, zns_center),
        size=mp.Vector3(total_width, mp.inf, zns_thick),
        material=materials['zns']
    ))
    
    # Ge 底层
    geometry.append(mp.Block(
        center=mp.Vector3(0, 0, ge_bottom_center),
        size=mp.Vector3(total_width, mp.inf, ge_bottom_thick),
        material=materials['ge']
    ))
    
    # Au 反射层
    geometry.append(mp.Block(
        center=mp.Vector3(0, 0, au_center),
        size=mp.Vector3(total_width, mp.inf, au_thick),
        material=materials['au']
    ))
    
    # 创建多周期的 NURBS 纳米线
    for period_idx in range(num_periods):
        x_offset = (period_idx - (num_periods - 1) / 2) * period_x
        
        for curve_idx, (points, width) in enumerate(zip(control_points, line_widths)):
            # 偏移控制点
            points_offset = [[p[0] + x_offset, p[1]] for p in points]
            
            # 创建 NURBS 棱镜
            prism = create_nurbs_prism(
                points_offset, 
                line_width=width,
                height=ag_height,
                z_center=ag_height / 2,
                material=materials['ag']
            )
            if prism is not None:
                geometry.append(prism)
    
    return geometry, total_width


class AngleSweepSimulation:
    """角度扫描仿真类"""
    
    def __init__(self, control_points, line_widths, resolution=10, num_periods=5):
        """
        初始化角度扫描仿真
        
        Args:
            control_points: NURBS 控制点
            line_widths: 线宽列表
            resolution: 网格分辨率
            num_periods: x 方向周期数
        """
        self.control_points = control_points
        self.line_widths = line_widths
        self.resolution = resolution
        self.num_periods = num_periods
        
        # 结构参数
        self.period_x = 12.6  # um
        self.period_y = 12.6  # um
        
        # 计算总宽度
        self.total_width = num_periods * self.period_x
        
        # PML 和边距
        self.pml_thickness = 2.0  # PML 厚度
        self.z_padding = 3.0  # z 方向边距
        
        # 波长范围
        self.wavelength_min = 11.6  # um
        self.wavelength_max = 13.6  # um
        self.wavelength_center = (self.wavelength_min + self.wavelength_max) / 2
        
        # 频率
        self.fcen = 1 / self.wavelength_center
        self.df = 1/self.wavelength_min - 1/self.wavelength_max
        self.nfreq = 101  # 频率点数
        
        # 材料
        self.materials = {
            'ge': create_ge_material_palik(),
            'zns': create_zns_material_from_data(),
            'au': create_au_material_stable(resolution),
            'ag': create_ag_material_stable(resolution),
        }
    
    def run_single_angle(self, angle_deg, sim_time=300, output_prefix="angle"):
        """
        运行单个角度的仿真
        
        Args:
            angle_deg: 入射角度 (度)
            sim_time: 仿真时间
            output_prefix: 输出文件前缀
        
        Returns:
            freqs: 频率数组
            reflectance: 反射率数组
        """
        angle_rad = np.radians(angle_deg)
        
        # 计算仿真区域尺寸
        # z 方向: 层结构 + 边距 + PML
        z_min = -0.85 - self.z_padding - self.pml_thickness
        z_max = self.z_padding + self.pml_thickness
        sz = z_max - z_min
        
        # x 方向: 多周期 + PML (两端)
        sx = self.total_width + 2 * self.pml_thickness
        
        # y 方向: 单周期 (周期边界)
        sy = self.period_y
        
        cell_size = mp.Vector3(sx, sy, sz)
        
        # 创建几何结构
        geometry, _ = create_multi_period_geometry(
            self.control_points, self.line_widths,
            num_periods=self.num_periods,
            period_x=self.period_x,
            materials=self.materials
        )
        
        # PML 边界条件
        # x 方向: PML
        # y 方向: 周期
        # z 方向: PML
        pml_layers = [
            mp.PML(thickness=self.pml_thickness, direction=mp.X),
            mp.PML(thickness=self.pml_thickness, direction=mp.Z),
        ]
        
        # 光源设置 - 倾斜入射高斯光束
        # k_point 用于设置入射角度
        # kx = sin(theta) * n * fcen, ky = 0
        n_air = 1.0
        kx = np.sin(angle_rad) * n_air * self.fcen
        k_point = mp.Vector3(kx, 0, 0)
        
        # 光源位置 (z 方向)
        src_z = z_max - self.pml_thickness - 1.0
        
        # 使用高斯光束源，宽度覆盖多周期
        sources = [
            mp.Source(
                mp.GaussianSource(self.fcen, fwidth=self.df),
                component=mp.Ex,  # TE 极化
                center=mp.Vector3(0, 0, src_z),
                size=mp.Vector3(self.total_width, sy, 0),
            )
        ]
        
        # 创建仿真对象
        sim = mp.Simulation(
            cell_size=cell_size,
            geometry=geometry,
            sources=sources,
            resolution=self.resolution,
            boundary_layers=pml_layers,
            k_point=k_point,  # 设置倾斜入射
        )
        
        # 反射监测器
        refl_z = src_z - 0.5  # 略低于光源
        refl_flux = sim.add_flux(
            self.fcen, self.df, self.nfreq,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, refl_z),
                size=mp.Vector3(self.total_width, sy, 0)
            )
        )
        
        # 运行仿真
        if mp.am_master():
            print(f"\n{'='*60}")
            print(f"Running angle = {angle_deg}° (k_x = {kx:.4f})")
            print(f"{'='*60}")
        
        sim.run(until=sim_time)
        
        # 获取反射通量
        flux = mp.get_fluxes(refl_flux)
        freqs = mp.get_flux_freqs(refl_flux)
        
        # 清理
        sim.reset_meep()
        
        return np.array(freqs), np.array(flux)
    
    def run_angle_sweep(self, angles, sim_time=300, output_dir="sweep_results"):
        """
        运行多角度扫描
        
        Args:
            angles: 角度列表 (度)
            sim_time: 每个角度的仿真时间
            output_dir: 输出目录
        
        Returns:
            results: 字典 {angle: (freqs, flux)}
        """
        if mp.am_master():
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n{'#'*60}")
            print(f"# Angle Sweep Simulation")
            print(f"# Angles: {angles}")
            print(f"# Periods: {self.num_periods}")
            print(f"# Resolution: {self.resolution}")
            print(f"{'#'*60}\n")
        
        results = {}
        
        for angle in angles:
            freqs, flux = self.run_single_angle(angle, sim_time)
            results[angle] = (freqs, flux)
            
            # 保存单个角度结果
            if mp.am_master():
                wavelengths = 1.0 / freqs
                output_file = os.path.join(output_dir, f"angle_{angle:05.1f}.csv")
                np.savetxt(output_file, 
                          np.column_stack([wavelengths, flux]),
                          delimiter=',',
                          header='wavelength_um,flux',
                          comments='')
                print(f"Saved: {output_file}")
        
        # 保存汇总结果
        if mp.am_master():
            self._save_summary(results, output_dir)
            self._plot_dispersion(results, output_dir)
        
        return results
    
    def _save_summary(self, results, output_dir):
        """保存汇总数据"""
        angles = sorted(results.keys())
        freqs = results[angles[0]][0]
        wavelengths = 1.0 / freqs
        
        # 创建 2D 数据矩阵
        flux_matrix = np.zeros((len(wavelengths), len(angles)))
        for i, angle in enumerate(angles):
            flux_matrix[:, i] = results[angle][1]
        
        # 保存为 NPZ 格式
        np.savez(
            os.path.join(output_dir, "sweep_summary.npz"),
            wavelengths=wavelengths,
            angles=np.array(angles),
            flux=flux_matrix
        )
        
        # 也保存为 CSV
        header = "wavelength_um," + ",".join([f"angle_{a:.1f}" for a in angles])
        data = np.column_stack([wavelengths, flux_matrix])
        np.savetxt(
            os.path.join(output_dir, "sweep_summary.csv"),
            data, delimiter=',', header=header, comments=''
        )
    
    def _plot_dispersion(self, results, output_dir):
        """绘制色散图/能带图"""
        import matplotlib.pyplot as plt
        
        angles = sorted(results.keys())
        freqs = results[angles[0]][0]
        wavelengths = 1.0 / freqs
        
        # 创建 2D 数据矩阵
        flux_matrix = np.zeros((len(wavelengths), len(angles)))
        for i, angle in enumerate(angles):
            flux_matrix[:, i] = results[angle][1]
        
        # 归一化
        flux_norm = np.abs(flux_matrix)
        flux_norm = flux_norm / np.max(flux_norm)
        
        # 计算反射率（假设归一化）
        reflectance = flux_norm
        
        # 发射率 = 1 - R
        emissivity = 1 - reflectance
        
        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 反射率色散图
        ax1 = axes[0]
        X, Y = np.meshgrid(angles, wavelengths)
        im1 = ax1.pcolormesh(X, Y, reflectance, shading='auto', cmap='hot_r')
        plt.colorbar(im1, ax=ax1, label='Reflectance')
        ax1.set_xlabel('Incident Angle (°)', fontsize=12)
        ax1.set_ylabel('Wavelength (μm)', fontsize=12)
        ax1.set_title('Reflectance Dispersion', fontsize=14)
        
        # 发射率色散图
        ax2 = axes[1]
        im2 = ax2.pcolormesh(X, Y, emissivity, shading='auto', cmap='hot')
        plt.colorbar(im2, ax=ax2, label='Emissivity')
        ax2.set_xlabel('Incident Angle (°)', fontsize=12)
        ax2.set_ylabel('Wavelength (μm)', fontsize=12)
        ax2.set_title('Emissivity Dispersion (Band-like)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dispersion_map.png"), dpi=200)
        plt.close()
        
        print(f"Dispersion map saved to {output_dir}/dispersion_map.png")


def main():
    parser = argparse.ArgumentParser(description='Angle sweep simulation for nonlocal metasurface')
    parser.add_argument('--angles', type=float, nargs='+', default=[0, 10, 20, 30, 40, 50, 60],
                        help='Incident angles in degrees')
    parser.add_argument('--resolution', type=int, default=10,
                        help='Grid resolution (pixels/um)')
    parser.add_argument('--periods', type=int, default=5,
                        help='Number of periods in x direction')
    parser.add_argument('--time', type=int, default=300,
                        help='Simulation time per angle')
    parser.add_argument('--output', type=str, default='angle_sweep_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # 控制点定义 - 与 Lumerical 一致
    d = np.array([0.2, 0.3, 0.5, 0.7, 1.0, 1.0])
    l = np.array([5.0, 4.8, 4.6, 3.5, 3.0, 3.0])
    x_centers = [-5.25, -3.15, -1.05, 1.05, 3.15, 5.25]
    
    control_points = [
        [[x_centers[i] + d[i], -l[i]], [x_centers[i], 0], [x_centers[i] + d[i], l[i]]]
        for i in range(6)
    ]
    line_widths = [1.1, 1.0, 0.8, 0.7, 0.6, 0.5]
    
    # 创建仿真对象
    sim = AngleSweepSimulation(
        control_points, line_widths,
        resolution=args.resolution,
        num_periods=args.periods
    )
    
    # 运行角度扫描
    results = sim.run_angle_sweep(
        angles=args.angles,
        sim_time=args.time,
        output_dir=args.output
    )
    
    if mp.am_master():
        print("\n" + "="*60)
        print("Angle sweep completed!")
        print(f"Results saved to: {args.output}/")
        print("="*60)


if __name__ == "__main__":
    main()
