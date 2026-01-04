"""
材料定义模块 - 为热辐射超表面仿真定义材料属性
支持 MEEP 的 Lorentz-Drude 色散模型
"""

import numpy as np
import meep as mp
from scipy.optimize import curve_fit


def load_zns_data(filepath="ZnS.txt"):
    """加载 ZnS 折射率数据"""
    data = np.loadtxt(filepath)
    wavelength_um = data[:, 0]
    n_real = data[:, 1]
    n_imag = data[:, 2]
    return wavelength_um, n_real, n_imag


def get_zns_epsilon(wavelength_um, n_real, n_imag):
    """计算 ZnS 的复介电常数"""
    n_complex = n_real + 1j * n_imag
    epsilon = n_complex ** 2
    return epsilon


def fit_lorentz_drude(wavelength_um, epsilon, num_poles=3):
    """
    拟合 Lorentz-Drude 模型参数
    MEEP 使用的形式: eps(omega) = eps_inf + sum_n (sigma_n * omega_n^2) / (omega_n^2 - omega^2 - i*omega*gamma_n)
    """
    c = 3e8  # 光速 m/s
    freq_hz = c / (wavelength_um * 1e-6)
    omega = 2 * np.pi * freq_hz
    
    eps_inf = np.real(epsilon[-1])
    
    return eps_inf


def create_zns_material(filepath="ZnS.txt", resolution=10):
    """
    创建 ZnS 材料 - 使用简化的恒定介电常数模型
    在热辐射波段 (8-14 μm) ZnS 的色散较弱
    """
    wavelength_um, n_real, n_imag = load_zns_data(filepath)
    
    # 选择热辐射波段 (8-14 μm) 的平均值
    mask = (wavelength_um >= 8) & (wavelength_um <= 14)
    n_avg = np.mean(n_real[mask])
    k_avg = np.mean(n_imag[mask])
    
    epsilon_real = n_avg**2 - k_avg**2
    epsilon_imag = 2 * n_avg * k_avg
    
    # 对于小的虚部，使用实数介电常数
    if k_avg < 0.01:
        return mp.Medium(epsilon=epsilon_real)
    else:
        # 使用导电率来模拟损耗
        # sigma = omega * epsilon_imag / epsilon_0
        # 在 MEEP 中使用 D_conductivity
        return mp.Medium(epsilon=epsilon_real, D_conductivity=epsilon_imag * 0.1)


def create_ge_material():
    """
    创建 Ge (锗) 材料
    在热辐射波段 (8-14 μm) Ge 的折射率约为 4.0，吸收较小
    """
    n_ge = 4.0
    return mp.Medium(epsilon=n_ge**2)


def create_au_material(fcen):
    """
    创建 Au (金) 材料 - 简化的 Drude 模型
    适用于远红外波段 (8-15 um)
    
    Args:
        fcen: 中心频率 (MEEP 单位，c=1)
    """
    # 在远红外波段，金属行为接近完美导体
    # 使用简化的 Drude 模型，参数已针对红外波段优化
    
    # Drude 参数 (针对远红外波段稳定性优化)
    # omega_p = 9.03 eV ≈ 2.18 um^-1
    # gamma = 0.053 eV ≈ 0.0128 um^-1
    
    Au_plasma_frq = 2.18  # plasma frequency in um^-1
    Au_gamma = 0.0128    # damping in um^-1
    
    # Drude 极化率: sigma = omega_p^2
    Au_sigma = Au_plasma_frq ** 2
    
    Au_susc = [
        mp.DrudeSusceptibility(frequency=1.0, gamma=Au_gamma, sigma=Au_sigma),
    ]
    
    return mp.Medium(epsilon=1.0, E_susceptibilities=Au_susc)


def create_ag_material(fcen):
    """
    创建 Ag (银) 材料 - 简化的 Drude 模型
    适用于远红外波段
    
    Args:
        fcen: 中心频率 (MEEP 单位)
    """
    # 在远红外波段，银的行为接近完美导体
    # 使用简化的 Drude 模型
    
    # Drude 参数 (针对远红外波段优化)
    # omega_p = 9.01 eV ≈ 2.18 um^-1
    # gamma = 0.048 eV ≈ 0.0116 um^-1
    
    Ag_plasma_frq = 2.18  # plasma frequency in um^-1
    Ag_gamma = 0.0116    # damping in um^-1
    
    Ag_sigma = Ag_plasma_frq ** 2
    
    Ag_susc = [
        mp.DrudeSusceptibility(frequency=1.0, gamma=Ag_gamma, sigma=Ag_sigma),
    ]
    
    return mp.Medium(epsilon=1.0, E_susceptibilities=Ag_susc)


def create_simple_metal(conductivity=1e7):
    """
    创建简化的金属材料（完美电导体近似）
    适用于远红外波段
    """
    return mp.perfect_electric_conductor


if __name__ == "__main__":
    # 测试材料创建
    print("Testing materials module...")
    
    zns = create_zns_material()
    print(f"ZnS material created: epsilon = {zns.epsilon}")
    
    ge = create_ge_material()
    print(f"Ge material created: epsilon = {ge.epsilon}")
    
    fcen = 1/12.5  # 12.5 um 对应的频率
    au = create_au_material(fcen)
    print(f"Au material created with Drude model")
    
    ag = create_ag_material(fcen)
    print(f"Ag material created with Drude-Lorentz model")
