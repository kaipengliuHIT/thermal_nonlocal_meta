#!/usr/bin/env python3
"""
高效并行角度扫描 - 充分利用多核心

策略：同时运行多个角度的仿真，每个角度使用部分核心
例如：128核心 = 8个并行任务 × 16核心/任务

运行方式:
    python parallel_angle_sweep.py --angles-start 0 --angles-end 60 --angles-step 5 \
        --total-cores 128 --cores-per-job 16 --resolution 12 --time 300
"""

import numpy as np
import subprocess
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time


def run_single_angle_job(args):
    """运行单个角度的仿真任务"""
    angle, cores, resolution, periods, sim_time, output_dir, reference = args
    
    prefix = "ref" if reference else "angle"
    output_file = os.path.join(output_dir, f"{prefix}_{angle:05.1f}.csv")
    log_file = os.path.join(output_dir, f"{prefix}_{angle:05.1f}.log")
    
    # 构建 MEEP 命令
    cmd = [
        "mpirun", "-np", str(cores),
        "python", "angle_sweep_single.py",
        "--angle", str(angle),
        "--resolution", str(resolution),
        "--periods", str(periods),
        "--time", str(sim_time),
        "--output", output_dir
    ]
    if reference:
        cmd.append("--reference")
    
    mode_str = "REF" if reference else "SIM"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {mode_str} angle={angle}° with {cores} cores")
    
    start_time = time.time()
    
    with open(log_file, 'w') as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Completed {mode_str} angle={angle}° in {elapsed:.1f}s")
        return (angle, True, elapsed, reference)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed {mode_str} angle={angle}°")
        return (angle, False, elapsed, reference)


def generate_dispersion_plot(output_dir, angles):
    """生成高质量色散图"""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # 收集数据
    all_data = []
    valid_angles = []
    
    for angle in sorted(angles):
        csv_file = os.path.join(output_dir, f"angle_{angle:05.1f}.csv")
        if os.path.exists(csv_file):
            data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
            all_data.append(data)
            valid_angles.append(angle)
    
    if len(all_data) < 2:
        print("Not enough data for dispersion plot")
        return
    
    # 构建 2D 矩阵
    wavelengths = all_data[0][:, 0]
    flux_matrix = np.zeros((len(wavelengths), len(valid_angles)))
    
    for i, data in enumerate(all_data):
        flux_matrix[:, i] = np.abs(data[:, 1])
    
    # 归一化
    flux_max = np.max(flux_matrix)
    if flux_max > 0:
        flux_norm = flux_matrix / flux_max
    else:
        flux_norm = flux_matrix
    
    # 计算发射率
    reflectance = flux_norm
    emissivity = 1 - reflectance
    emissivity = np.clip(emissivity, 0, 1)
    
    # 创建高质量图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    X, Y = np.meshgrid(valid_angles, wavelengths)
    
    # 反射率色散图
    ax1 = axes[0]
    im1 = ax1.pcolormesh(X, Y, reflectance, shading='gouraud', cmap='hot_r', 
                         vmin=0, vmax=1)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Reflectance', fontsize=12)
    ax1.set_xlabel('Incident Angle (°)', fontsize=14)
    ax1.set_ylabel('Wavelength (μm)', fontsize=14)
    ax1.set_title('Reflectance Dispersion', fontsize=16)
    ax1.tick_params(labelsize=12)
    
    # 吸收率色散图（类似能带图）- 使用蓝色色带匹配 Lumerical 风格
    ax2 = axes[1]
    im2 = ax2.pcolormesh(X, Y, emissivity, shading='gouraud', cmap='Blues',
                         vmin=0, vmax=1)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Absorption (A = 1 - R)', fontsize=12)
    ax2.set_xlabel('Incident Angle (°)', fontsize=14)
    ax2.set_ylabel('Wavelength (μm)', fontsize=14)
    ax2.set_title('Absorption Dispersion', fontsize=16)
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dispersion_map_hires.png"), dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    # 保存汇总数据
    np.savez(
        os.path.join(output_dir, "sweep_data.npz"),
        wavelengths=wavelengths,
        angles=np.array(valid_angles),
        reflectance=reflectance,
        emissivity=emissivity
    )
    
    print(f"\nDispersion map saved: {output_dir}/dispersion_map_hires.png")


def main():
    parser = argparse.ArgumentParser(
        description='Parallel angle sweep for nonlocal metasurface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage for 128-core system:
    python parallel_angle_sweep.py --angles-start 0 --angles-end 60 --angles-step 2.5 \\
        --total-cores 128 --cores-per-job 16 --resolution 12 --time 300

This will run 8 jobs in parallel (128/16=8), each using 16 MPI processes.
        """
    )
    
    parser.add_argument('--angles-start', type=float, default=0,
                        help='Starting angle (degrees)')
    parser.add_argument('--angles-end', type=float, default=60,
                        help='Ending angle (degrees)')
    parser.add_argument('--angles-step', type=float, default=5,
                        help='Angle step size (degrees)')
    parser.add_argument('--total-cores', type=int, default=128,
                        help='Total number of CPU cores available')
    parser.add_argument('--cores-per-job', type=int, default=16,
                        help='Number of cores per MEEP job')
    parser.add_argument('--resolution', type=int, default=12,
                        help='Grid resolution (pixels/um)')
    parser.add_argument('--periods', type=int, default=5,
                        help='Number of periods in x direction')
    parser.add_argument('--time', type=int, default=300,
                        help='Simulation time per angle')
    parser.add_argument('--output', type=str, default='parallel_sweep_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # 生成角度列表
    angles = np.arange(args.angles_start, args.angles_end + args.angles_step/2, 
                       args.angles_step)
    
    # 计算并行任务数
    max_parallel = args.total_cores // args.cores_per_job
    
    print("=" * 70)
    print("PARALLEL ANGLE SWEEP SIMULATION")
    print("=" * 70)
    print(f"Angles: {angles[0]}° to {angles[-1]}° (step {args.angles_step}°)")
    print(f"Total angles: {len(angles)}")
    print(f"Total cores: {args.total_cores}")
    print(f"Cores per job: {args.cores_per_job}")
    print(f"Max parallel jobs: {max_parallel}")
    print(f"Resolution: {args.resolution} pixels/μm")
    print(f"Periods: {args.periods}")
    print(f"Sim time: {args.time}")
    print(f"Output: {args.output}/")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 准备任务参数 - 同时运行参考仿真和实际仿真
    job_args = []
    for angle in angles:
        # 实际仿真（包含纳米线）
        job_args.append(
            (angle, args.cores_per_job, args.resolution, args.periods, 
             args.time, args.output, False)
        )
    
    # 使用进程池并行执行
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(run_single_angle_job, arg): arg[0] 
                   for arg in job_args}
        
        for future in as_completed(futures):
            angle = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error for angle {angle}°: {e}")
                results.append((angle, False, 0, False))
    
    total_time = time.time() - start_time
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results if len(r) >= 2 and r[1])
    print(f"Completed: {successful}/{len(angles)} angles")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    if successful > 0:
        avg_time = sum(r[2] for r in results if r[1]) / successful
        print(f"Average time per angle: {avg_time:.1f}s")
        print(f"Speedup from parallelization: {len(angles) * avg_time / total_time:.1f}x")
    
    # 生成色散图
    if successful > 1:
        print("\nGenerating dispersion plot...")
        generate_dispersion_plot(args.output, angles)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
