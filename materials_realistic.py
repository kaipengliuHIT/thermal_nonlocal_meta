"""
真实材料参数模块 - 基于 Palik 数据库
为热辐射超表面仿真 (8-15 μm 波段) 提供材料定义

材料数据来源:
- Au, Ag: Rakic et al. (1998) Applied Optics 37, 5271-5283
- Ge: Palik Handbook of Optical Constants
- ZnS: 用户提供的实验数据

MEEP 单位说明:
- 长度单位: μm (微米)
- 频率单位: c/μm (其中 c=1)
- 1 eV = 0.2418 μm^-1 (频率)
"""

import numpy as np
import meep as mp
from scipy.optimize import minimize


# 单位转换常数
eV_to_um_freq = 0.2418  # 1 eV 对应的 MEEP 频率 (um^-1)


def create_au_material_palik():
    """
    创建 Au (金) 材料 - 基于 Rakic 的 Drude-Lorentz 模型
    参数来源: Rakic et al. Applied Optics 37, 5271 (1998)
    
    该模型在 0.2-20 μm 波段有效
    """
    # Rakic 的 Drude-Lorentz 模型参数 (已转换为 MEEP 单位)
    # eps(w) = eps_inf - f0*wp^2/(w^2+i*g0*w) + sum_j fj*wp^2/(wj^2-w^2-i*gj*w)
    
    # 等离子体频率
    wp = 9.03 * eV_to_um_freq  # 2.183 um^-1
    
    # Drude 项 (自由电子)
    f0 = 0.760
    g0 = 0.053 * eV_to_um_freq  # 0.0128 um^-1
    
    # Lorentz 项 (带间跃迁) - 对于远红外主要是 Drude 项
    # 但为了精确性，包含主要的 Lorentz 项
    f1 = 0.024
    w1 = 0.415 * eV_to_um_freq
    g1 = 0.241 * eV_to_um_freq
    
    f2 = 0.010
    w2 = 0.830 * eV_to_um_freq
    g2 = 0.345 * eV_to_um_freq
    
    # 在 MEEP 中，Drude 极化率形式:
    # chi = sigma * w0^2 / (w0^2 - w^2 - i*w*gamma)
    # 对于 Drude (w0→0): chi = -sigma / (w^2 + i*w*gamma) = sigma / (i*w*gamma - w^2)
    
    # MEEP DrudeSusceptibility: sigma * frequency^2 / (frequency^2 - w^2 - i*w*gamma)
    # 当 frequency→0 时变为纯 Drude
    
    Au_susc = [
        # Drude 项: 使用 DrudeSusceptibility
        mp.DrudeSusceptibility(frequency=1.0, gamma=g0, sigma=f0 * wp**2),
        # Lorentz 项
        mp.LorentzianSusceptibility(frequency=w1, gamma=g1, sigma=f1 * wp**2 / w1**2),
        mp.LorentzianSusceptibility(frequency=w2, gamma=g2, sigma=f2 * wp**2 / w2**2),
    ]
    
    return mp.Medium(epsilon=1.0, E_susceptibilities=Au_susc)


def create_ag_material_palik():
    """
    创建 Ag (银) 材料 - 基于 Rakic 的 Drude-Lorentz 模型
    参数来源: Rakic et al. Applied Optics 37, 5271 (1998)
    
    该模型在 0.2-20 μm 波段有效
    """
    # 等离子体频率
    wp = 9.01 * eV_to_um_freq  # 2.178 um^-1
    
    # Drude 项
    f0 = 0.845
    g0 = 0.048 * eV_to_um_freq  # 0.0116 um^-1
    
    # Lorentz 项
    f1 = 0.065
    w1 = 0.816 * eV_to_um_freq
    g1 = 3.886 * eV_to_um_freq
    
    f2 = 0.124
    w2 = 4.481 * eV_to_um_freq
    g2 = 0.452 * eV_to_um_freq
    
    Ag_susc = [
        mp.DrudeSusceptibility(frequency=1.0, gamma=g0, sigma=f0 * wp**2),
        mp.LorentzianSusceptibility(frequency=w1, gamma=g1, sigma=f1 * wp**2 / w1**2),
        mp.LorentzianSusceptibility(frequency=w2, gamma=g2, sigma=f2 * wp**2 / w2**2),
    ]
    
    return mp.Medium(epsilon=1.0, E_susceptibilities=Ag_susc)


def create_ge_material_palik():
    """
    创建 Ge (锗) 材料 - 基于 Palik 数据
    
    在远红外波段 (8-15 μm)，Ge 的折射率约为 4.0，几乎无吸收
    使用 Sellmeier 方程的简化形式
    """
    # Ge 在红外的参数 (Palik)
    # n ≈ 4.0, k ≈ 0 (在 8-15 μm)
    # 使用简单的实数介电常数
    n_ge = 4.003
    
    # 也可以加入轻微的色散
    # Sellmeier 系数 (简化)
    A = 16.0  # n^2 ≈ 16
    
    return mp.Medium(epsilon=A)


def create_zns_material_from_data(filepath="ZnS.txt", wavelength_range=(8, 15)):
    """
    从实验数据创建 ZnS 材料
    
    Args:
        filepath: ZnS 数据文件路径
        wavelength_range: 感兴趣的波长范围 (μm)
    
    Returns:
        mp.Medium 对象
    """
    try:
        data = np.loadtxt(filepath)
        wavelength_um = data[:, 0]
        n_real = data[:, 1]
        n_imag = data[:, 2]
        
        # 选择目标波长范围
        mask = (wavelength_um >= wavelength_range[0]) & (wavelength_um <= wavelength_range[1])
        
        if np.sum(mask) > 0:
            # 使用该范围的平均值
            n_avg = np.mean(n_real[mask])
            k_avg = np.mean(n_imag[mask])
        else:
            # 如果没有数据在范围内，使用整体平均
            n_avg = np.mean(n_real)
            k_avg = np.mean(n_imag)
        
        epsilon_real = n_avg**2 - k_avg**2
        epsilon_imag = 2 * n_avg * k_avg
        
        print(f"ZnS material: n = {n_avg:.4f}, k = {k_avg:.4f}")
        print(f"  epsilon = {epsilon_real:.4f} + {epsilon_imag:.4f}i")
        
        if k_avg < 0.01:
            return mp.Medium(epsilon=epsilon_real)
        else:
            # 使用导电率模拟损耗
            # D_conductivity 在 MEEP 单位中
            return mp.Medium(epsilon=epsilon_real, D_conductivity=epsilon_imag)
            
    except Exception as e:
        print(f"Warning: Could not load ZnS data from {filepath}: {e}")
        print("Using default ZnS parameters")
        # 默认参数
        return mp.Medium(epsilon=5.2)


def create_au_material_stable(resolution=10):
    """
    创建数值稳定的 Au 材料 - 适用于低分辨率仿真
    
    对于远红外波段，金属的响应主要由 Drude 项决定
    简化模型以提高数值稳定性
    
    Args:
        resolution: 网格分辨率，用于调整参数
    """
    # 在远红外 (10-15 μm)，金的介电函数近似为:
    # eps ≈ -10000 + i*1000 (强金属响应)
    
    # 使用高导电率近似
    # eps = eps_inf - sigma / (i * w) ≈ eps_inf + i * sigma / w
    
    # 简化的 Drude 参数
    wp = 2.18  # 等离子体频率 (um^-1)
    gamma = 0.02  # 阻尼 (um^-1)，稍微增大以提高稳定性
    
    # Drude sigma
    sigma_d = wp ** 2
    
    Au_susc = [
        mp.DrudeSusceptibility(frequency=1.0, gamma=gamma, sigma=sigma_d),
    ]
    
    return mp.Medium(epsilon=1.0, E_susceptibilities=Au_susc)


def create_ag_material_stable(resolution=10):
    """
    创建数值稳定的 Ag 材料 - 适用于低分辨率仿真
    """
    wp = 2.18  # 等离子体频率 (um^-1)
    gamma = 0.02  # 阻尼 (um^-1)
    
    sigma_d = wp ** 2
    
    Ag_susc = [
        mp.DrudeSusceptibility(frequency=1.0, gamma=gamma, sigma=sigma_d),
    ]
    
    return mp.Medium(epsilon=1.0, E_susceptibilities=Ag_susc)


def create_material_with_conductivity(sigma_dc, epsilon_inf=1.0):
    """
    使用直流电导率创建金属材料
    适用于远红外波段的简化模型
    
    Args:
        sigma_dc: 直流电导率 (S/m)，Au ≈ 4.1e7, Ag ≈ 6.3e7
        epsilon_inf: 高频介电常数
    """
    # 在 MEEP 中，D_conductivity 的单位是 (MEEP频率单位)
    # 需要转换: D_conductivity = sigma_dc / (eps_0 * 2*pi*c/um)
    
    eps_0 = 8.854e-12  # F/m
    c = 3e8  # m/s
    um = 1e-6
    
    # MEEP conductivity
    meep_conductivity = sigma_dc / (eps_0 * 2 * np.pi * c / um)
    
    return mp.Medium(epsilon=epsilon_inf, D_conductivity=meep_conductivity)


def get_metal_epsilon_at_wavelength(material_type, wavelength_um):
    """
    计算指定波长处的金属介电常数（用于验证）
    
    Args:
        material_type: 'Au' 或 'Ag'
        wavelength_um: 波长 (μm)
    
    Returns:
        复介电常数
    """
    freq = 1.0 / wavelength_um  # MEEP 频率
    omega = 2 * np.pi * freq
    
    if material_type == 'Au':
        wp = 9.03 * eV_to_um_freq
        f0, g0 = 0.760, 0.053 * eV_to_um_freq
    else:  # Ag
        wp = 9.01 * eV_to_um_freq
        f0, g0 = 0.845, 0.048 * eV_to_um_freq
    
    # Drude 响应
    eps_drude = -f0 * wp**2 / (omega**2 + 1j * g0 * omega)
    
    return 1.0 + eps_drude


if __name__ == "__main__":
    print("Testing realistic materials module...")
    print("=" * 50)
    
    # 测试金属在远红外的介电常数
    for wl in [10, 12, 14]:
        eps_au = get_metal_epsilon_at_wavelength('Au', wl)
        eps_ag = get_metal_epsilon_at_wavelength('Ag', wl)
        print(f"\nAt λ = {wl} μm:")
        print(f"  Au: ε = {eps_au.real:.1f} + {eps_au.imag:.1f}i")
        print(f"  Ag: ε = {eps_ag.real:.1f} + {eps_ag.imag:.1f}i")
    
    print("\n" + "=" * 50)
    print("Creating MEEP materials...")
    
    au = create_au_material_stable()
    print(f"Au (stable): created")
    
    ag = create_ag_material_stable()
    print(f"Ag (stable): created")
    
    ge = create_ge_material_palik()
    print(f"Ge: epsilon = {ge.epsilon}")
    
    zns = create_zns_material_from_data()
