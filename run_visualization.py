"""
运行仿真并生成完整可视化结果
包括：几何结构、电场分布、吸收谱
"""

import numpy as np
import meep as mp
import argparse
import os

from thermal_meta_sim import ThermalMetaSim
from visualization import (
    plot_geometry_2d, 
    plot_field_distribution,
    plot_epsilon_distribution,
    plot_spectrum,
    plot_spectrum_simple,
    generate_all_visualizations
)


def run_with_visualization(resolution=10, sim_time=200, output_dir="figures"):
    """
    运行仿真并生成可视化
    """
    # 创建输出目录
    if mp.am_master():
        os.makedirs(output_dir, exist_ok=True)
    
    # 定义控制点和线宽
    d = np.array([0.2, 0.3, 0.5, 0.7, 1.0, 1.0])
    l = np.array([5.0, 4.8, 4.6, 3.5, 3.0, 3.0])
    
    control_points = [
        [[-5.25 + d[i], -l[i]], [-5.25 + 2.1*i, 0], [-5.25 + d[i], l[i]]]
        for i in range(6)
    ]
    line_widths = [1.1, 1.0, 0.8, 0.7, 0.6, 0.5]
    
    # 1. 生成静态几何可视化
    if mp.am_master():
        print("=" * 60)
        print("Step 1: Generating static geometry visualizations...")
        print("=" * 60)
        generate_all_visualizations(control_points, line_widths, output_dir)
    
    # 2. 创建仿真对象
    if mp.am_master():
        print("\n" + "=" * 60)
        print("Step 2: Setting up simulation...")
        print("=" * 60)
    
    sim_obj = ThermalMetaSim(control_points, line_widths, resolution=resolution)
    sim_obj.setup_simulation()
    
    # 3. 运行仿真
    if mp.am_master():
        print("\n" + "=" * 60)
        print(f"Step 3: Running simulation (time={sim_time})...")
        print("=" * 60)
    
    sim_obj.sim.run(until=sim_time)
    
    # 4. 获取反射通量并绘制光谱
    if mp.am_master():
        print("\n" + "=" * 60)
        print("Step 4: Calculating and plotting spectrum...")
        print("=" * 60)
    
    refl_flux = mp.get_fluxes(sim_obj.refl_flux)
    freqs = mp.get_flux_freqs(sim_obj.refl_flux)
    wavelengths = 1.0 / np.array(freqs)
    
    # 将通量转换为反射率（需要归一化）
    flux_array = np.array(refl_flux)
    reflectance = np.abs(flux_array) / np.max(np.abs(flux_array))
    
    # 保存数据
    if mp.am_master():
        output_file = os.path.join(output_dir, "spectrum_data.csv")
        np.savetxt(output_file, 
                   np.column_stack([wavelengths, flux_array, reflectance]),
                   delimiter=',',
                   header='wavelength_um,flux,reflectance_normalized',
                   comments='')
        print(f"Spectrum data saved to {output_file}")
    
    # 绘制光谱
    plot_spectrum(wavelengths, reflectance, 
                  output_file=os.path.join(output_dir, "absorption_spectrum.png"))
    
    if mp.am_master():
        print("\n" + "=" * 60)
        print("All visualizations completed!")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
        print("\nGenerated files:")
        for f in sorted(os.listdir(output_dir)):
            print(f"  - {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulation with visualization')
    parser.add_argument('--resolution', type=int, default=8,
                        help='Grid resolution (default: 8)')
    parser.add_argument('--time', type=int, default=150,
                        help='Simulation time (default: 150)')
    parser.add_argument('--output', type=str, default='figures',
                        help='Output directory (default: figures)')
    
    args = parser.parse_args()
    
    run_with_visualization(
        resolution=args.resolution,
        sim_time=args.time,
        output_dir=args.output
    )
