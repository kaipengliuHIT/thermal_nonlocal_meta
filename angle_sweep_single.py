"""
单角度仿真脚本 - 供并行扫描调用

运行方式:
    mpirun -np 16 python angle_sweep_single.py --angle 30 --resolution 12 --time 300
"""

import numpy as np
import meep as mp
import argparse
import os

from materials_realistic import (
    create_zns_material_from_data,
    create_ge_material_palik,
    create_au_material_palik,
    create_ag_material_palik,
)
from nurbs_geometry import create_nurbs_prism


def create_multi_period_geometry(control_points, line_widths, num_periods=5,
                                  period_x=12.6, materials=None):
    """创建多周期重复的几何结构"""
    geometry = []
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
    
    # 基底层
    geometry.append(mp.Block(
        center=mp.Vector3(0, 0, ge_top_center),
        size=mp.Vector3(total_width, mp.inf, ge_top_thick),
        material=materials['ge']
    ))
    
    geometry.append(mp.Block(
        center=mp.Vector3(0, 0, zns_center),
        size=mp.Vector3(total_width, mp.inf, zns_thick),
        material=materials['zns']
    ))
    
    geometry.append(mp.Block(
        center=mp.Vector3(0, 0, ge_bottom_center),
        size=mp.Vector3(total_width, mp.inf, ge_bottom_thick),
        material=materials['ge']
    ))
    
    geometry.append(mp.Block(
        center=mp.Vector3(0, 0, au_center),
        size=mp.Vector3(total_width, mp.inf, au_thick),
        material=materials['au']
    ))
    
    # 多周期 NURBS 纳米线
    for period_idx in range(num_periods):
        x_offset = (period_idx - (num_periods - 1) / 2) * period_x
        
        for curve_idx, (points, width) in enumerate(zip(control_points, line_widths)):
            points_offset = [[p[0] + x_offset, p[1]] for p in points]
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


def run_single_angle(angle_deg, resolution, num_periods, sim_time, output_dir, 
                     reference_mode=False):
    """
    运行单个角度的仿真
    
    Args:
        reference_mode: 如果为True，运行无结构的参考仿真用于归一化
    """
    
    # 控制点定义
    d = np.array([0.2, 0.3, 0.5, 0.7, 1.0, 1.0])
    l = np.array([5.0, 4.8, 4.6, 3.5, 3.0, 3.0])
    x_centers = [-5.25, -3.15, -1.05, 1.05, 3.15, 5.25]
    
    control_points = [
        [[x_centers[i] + d[i], -l[i]], [x_centers[i], 0], [x_centers[i] + d[i], l[i]]]
        for i in range(6)
    ]
    line_widths = [1.1, 1.0, 0.8, 0.7, 0.6, 0.5]
    
    # 结构参数
    period_x = 12.6
    period_y = 12.6
    total_width = num_periods * period_x
    pml_thickness = 1.0  # 减小 PML 厚度
    z_padding = 1.0  # 减小 z 方向边距，总高度约 4μm
    
    # 波长范围
    wavelength_min = 11.6
    wavelength_max = 13.6
    wavelength_center = (wavelength_min + wavelength_max) / 2
    fcen = 1 / wavelength_center
    df = 1/wavelength_min - 1/wavelength_max
    nfreq = 101
    
    # 材料 - 使用完整的 Palik/Rakic 数据库
    materials = {
        'ge': create_ge_material_palik(),
        'zns': create_zns_material_from_data(),
        'au': create_au_material_palik(),  # 完整 Drude-Lorentz 模型
        'ag': create_ag_material_palik(),  # 完整 Drude-Lorentz 模型
    }
    
    angle_rad = np.radians(angle_deg)
    
    # 仿真区域尺寸
    z_min = -0.85 - z_padding - pml_thickness
    z_max = z_padding + pml_thickness
    sz = z_max - z_min
    sx = total_width + 2 * pml_thickness
    sy = period_y
    
    cell_size = mp.Vector3(sx, sy, sz)
    
    # 几何结构
    if reference_mode:
        # 参考仿真：只有基底层，没有纳米线
        geometry = [
            mp.Block(
                center=mp.Vector3(0, 0, -0.05),
                size=mp.Vector3(total_width, mp.inf, 0.1),
                material=materials['ge']
            ),
            mp.Block(
                center=mp.Vector3(0, 0, -0.4),
                size=mp.Vector3(total_width, mp.inf, 0.6),
                material=materials['zns']
            ),
            mp.Block(
                center=mp.Vector3(0, 0, -0.75),
                size=mp.Vector3(total_width, mp.inf, 0.1),
                material=materials['ge']
            ),
            mp.Block(
                center=mp.Vector3(0, 0, -0.825),
                size=mp.Vector3(total_width, mp.inf, 0.05),
                material=materials['au']
            ),
        ]
    else:
        # 完整结构：包含纳米线
        geometry, _ = create_multi_period_geometry(
            control_points, line_widths,
            num_periods=num_periods,
            period_x=period_x,
            materials=materials
        )
    
    # PML 边界
    pml_layers = [
        mp.PML(thickness=pml_thickness, direction=mp.X),
        mp.PML(thickness=pml_thickness, direction=mp.Z),
    ]
    
    # 倾斜入射 k 点
    n_air = 1.0
    kx = np.sin(angle_rad) * n_air * fcen
    k_point = mp.Vector3(kx, 0, 0)
    
    # 光源
    src_z = z_max - pml_thickness - 1.0
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ex,
            center=mp.Vector3(0, 0, src_z),
            size=mp.Vector3(total_width, sy, 0),
        )
    ]
    
    # 创建仿真
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        boundary_layers=pml_layers,
        k_point=k_point,
    )
    
    # 反射监测器
    refl_z = src_z - 0.5
    refl_flux = sim.add_flux(
        fcen, df, nfreq,
        mp.FluxRegion(
            center=mp.Vector3(0, 0, refl_z),
            size=mp.Vector3(total_width, sy, 0)
        )
    )
    
    if mp.am_master():
        print(f"Running angle = {angle_deg}° (k_x = {kx:.4f})")
    
    # 运行
    sim.run(until=sim_time)
    
    # 获取结果
    flux = mp.get_fluxes(refl_flux)
    freqs = mp.get_flux_freqs(refl_flux)
    
    # 保存结果
    if mp.am_master():
        wavelengths = 1.0 / np.array(freqs)
        os.makedirs(output_dir, exist_ok=True)
        prefix = "ref_" if reference_mode else "angle_"
        output_file = os.path.join(output_dir, f"{prefix}{angle_deg:05.1f}.csv")
        np.savetxt(output_file, 
                  np.column_stack([wavelengths, flux]),
                  delimiter=',',
                  header='wavelength_um,flux',
                  comments='')
        print(f"Saved: {output_file}")
    
    sim.reset_meep()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--angle', type=float, required=True)
    parser.add_argument('--resolution', type=int, default=12)
    parser.add_argument('--periods', type=int, default=5)
    parser.add_argument('--time', type=int, default=300)
    parser.add_argument('--output', type=str, default='sweep_results')
    parser.add_argument('--reference', action='store_true',
                        help='Run reference simulation without nanowires')
    
    args = parser.parse_args()
    
    run_single_angle(
        args.angle, args.resolution, args.periods, 
        args.time, args.output, reference_mode=args.reference
    )


if __name__ == "__main__":
    main()
