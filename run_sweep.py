"""
参数扫描脚本 - 批量运行不同参数的仿真
支持 MPI 并行运行
"""

import numpy as np
import meep as mp
import os
import json
from datetime import datetime
from thermal_meta_sim import ThermalMetaSim


def sweep_line_width(base_control_points, width_range, resolution=10, until=300):
    """
    扫描纳米线线宽
    
    Args:
        base_control_points: 基础控制点
        width_range: 线宽范围 (start, stop, num)
        resolution: 网格分辨率
        until: 仿真时长
    """
    widths = np.linspace(*width_range)
    results = []
    
    for i, width in enumerate(widths):
        if mp.am_master():
            print(f"\n{'='*50}")
            print(f"Sweep {i+1}/{len(widths)}: line_width = {width:.2f} um")
            print(f"{'='*50}")
        
        line_widths = [width] * len(base_control_points)
        
        sim = ThermalMetaSim(
            control_points=base_control_points,
            line_widths=line_widths,
            resolution=resolution
        )
        
        wavelengths, reflectance, transmittance = sim.run_simulation(until=until)
        
        if mp.am_master():
            results.append({
                'line_width': width,
                'wavelengths': wavelengths.tolist(),
                'reflectance': reflectance.tolist(),
                'transmittance': transmittance.tolist()
            })
    
    return results


def sweep_period(base_control_points, period_range, resolution=10, until=300):
    """
    扫描周期
    
    Args:
        base_control_points: 基础控制点
        period_range: 周期范围 (start, stop, num)
    """
    periods = np.linspace(*period_range)
    results = []
    
    for i, period in enumerate(periods):
        if mp.am_master():
            print(f"\n{'='*50}")
            print(f"Sweep {i+1}/{len(periods)}: period = {period:.2f} um")
            print(f"{'='*50}")
        
        sim = ThermalMetaSim(
            control_points=base_control_points,
            resolution=resolution
        )
        sim.period_x = period
        sim.period_y = period
        
        wavelengths, reflectance, transmittance = sim.run_simulation(until=until)
        
        if mp.am_master():
            results.append({
                'period': period,
                'wavelengths': wavelengths.tolist(),
                'reflectance': reflectance.tolist(),
                'transmittance': transmittance.tolist()
            })
    
    return results


def sweep_curvature(d_range, l_range, resolution=10, until=300):
    """
    扫描曲线曲率参数
    
    Args:
        d_range: d 参数范围
        l_range: l 参数范围
    """
    d_values = np.linspace(*d_range)
    l_values = np.linspace(*l_range)
    results = []
    
    for i, d in enumerate(d_values):
        for j, l in enumerate(l_values):
            if mp.am_master():
                print(f"\n{'='*50}")
                print(f"Sweep: d = {d:.2f}, l = {l:.2f}")
                print(f"{'='*50}")
            
            # 构建控制点
            control_points = [
                [[-5.25 + d, -l], [-5.25, 0], [-5.25 + d, l]],
                [[-3.15 + d, -l], [-3.15, 0], [-3.15 + d, l]],
                [[-1.05 + d, -l], [-1.05, 0], [-1.05 + d, l]],
                [[1.05 + d, -l], [1.05, 0], [1.05 + d, l]],
                [[3.15 + d, -l], [3.15, 0], [3.15 + d, l]],
                [[5.25 + d, -l], [5.25, 0], [5.25 + d, l]]
            ]
            
            sim = ThermalMetaSim(
                control_points=control_points,
                resolution=resolution
            )
            
            wavelengths, reflectance, transmittance = sim.run_simulation(until=until)
            
            if mp.am_master():
                results.append({
                    'd': d,
                    'l': l,
                    'wavelengths': wavelengths.tolist(),
                    'reflectance': reflectance.tolist(),
                    'transmittance': transmittance.tolist()
                })
    
    return results


def save_sweep_results(results, filename):
    """保存扫描结果"""
    if mp.am_master():
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Sweep results saved to {filename}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parameter sweep for thermal metasurface')
    parser.add_argument('--sweep-type', type=str, choices=['width', 'period', 'curvature'],
                        default='width', help='Type of parameter sweep')
    parser.add_argument('--resolution', type=int, default=8,
                        help='Grid resolution')
    parser.add_argument('--until', type=float, default=300,
                        help='Simulation time')
    parser.add_argument('--output', type=str, default='sweep_results.json',
                        help='Output filename')
    args = parser.parse_args()
    
    # 基础控制点
    d = np.array([0.2, 0.3, 0.5, 0.7, 1.0, 1.0])
    l = np.array([5.0, 4.8, 4.6, 3.5, 3.0, 3.0])
    
    base_control_points = [
        [[-5.25 + d[i], -l[i]], [-5.25 + 2.1*i, 0], [-5.25 + d[i], l[i]]]
        for i in range(6)
    ]
    
    if args.sweep_type == 'width':
        if mp.am_master():
            print("Running line width sweep...")
        results = sweep_line_width(
            base_control_points,
            width_range=(0.3, 1.5, 5),
            resolution=args.resolution,
            until=args.until
        )
    
    elif args.sweep_type == 'period':
        if mp.am_master():
            print("Running period sweep...")
        results = sweep_period(
            base_control_points,
            period_range=(10.0, 15.0, 5),
            resolution=args.resolution,
            until=args.until
        )
    
    elif args.sweep_type == 'curvature':
        if mp.am_master():
            print("Running curvature sweep...")
        results = sweep_curvature(
            d_range=(0.1, 1.0, 3),
            l_range=(3.0, 5.0, 3),
            resolution=args.resolution,
            until=args.until
        )
    
    save_sweep_results(results, args.output)
    
    if mp.am_master():
        print("\nSweep completed!")


if __name__ == "__main__":
    main()
