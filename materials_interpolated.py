"""
材料数据插值拟合模块

MEEP 材料模型说明:
MEEP 使用时域有限差分 (FDTD) 方法，要求色散材料必须表示为极点形式 (Lorentz-Drude)，
这是因为时域中的卷积需要用辅助微分方程来实现。

但是，可以通过以下方法使用实验数据:
1. Vector Fitting: 从实验数据自动拟合 Lorentz 极点
2. 频域方法: 对于单频率仿真，可以使用插值的介电常数

本模块实现:
- 从实验数据自动拟合 Lorentz-Drude 极点
- 提供介电常数的插值函数用于验证
"""

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import minimize, differential_evolution
import meep as mp


def load_material_data(filepath):
    """
    加载材料数据文件
    
    Args:
        filepath: 数据文件路径，格式为 (wavelength_um, n, k)
    
    Returns:
        wavelength_um, n, k 数组
    """
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1], data[:, 2]


def nk_to_epsilon(n, k):
    """将 n, k 转换为复介电常数"""
    return (n + 1j * k) ** 2


def epsilon_to_nk(epsilon):
    """将复介电常数转换为 n, k"""
    n_complex = np.sqrt(epsilon)
    return np.real(n_complex), np.imag(n_complex)


class InterpolatedMaterial:
    """
    基于实验数据的插值材料类
    """
    
    def __init__(self, wavelength_um, n, k, kind='cubic'):
        """
        初始化插值材料
        
        Args:
            wavelength_um: 波长数组 (μm)
            n: 折射率实部
            k: 折射率虚部 (消光系数)
            kind: 插值类型 ('linear', 'cubic', 'quadratic')
        """
        self.wavelength_um = wavelength_um
        self.n = n
        self.k = k
        self.epsilon = nk_to_epsilon(n, k)
        
        # 创建插值函数
        self.n_interp = interp1d(wavelength_um, n, kind=kind, 
                                  bounds_error=False, fill_value='extrapolate')
        self.k_interp = interp1d(wavelength_um, k, kind=kind,
                                  bounds_error=False, fill_value='extrapolate')
        
        # 频率插值 (MEEP 使用频率)
        freq = 1.0 / wavelength_um  # MEEP 频率单位
        self.eps_real_interp = interp1d(freq, np.real(self.epsilon), kind=kind,
                                         bounds_error=False, fill_value='extrapolate')
        self.eps_imag_interp = interp1d(freq, np.imag(self.epsilon), kind=kind,
                                         bounds_error=False, fill_value='extrapolate')
    
    def get_nk(self, wavelength_um):
        """获取指定波长的 n, k 值"""
        return self.n_interp(wavelength_um), self.k_interp(wavelength_um)
    
    def get_epsilon(self, wavelength_um):
        """获取指定波长的复介电常数"""
        n, k = self.get_nk(wavelength_um)
        return nk_to_epsilon(n, k)
    
    def get_epsilon_at_freq(self, freq):
        """获取指定频率的复介电常数 (MEEP 频率单位)"""
        return self.eps_real_interp(freq) + 1j * self.eps_imag_interp(freq)


def lorentz_drude_epsilon(freq, eps_inf, poles):
    """
    计算 Lorentz-Drude 模型的介电常数
    
    Args:
        freq: 频率数组 (MEEP 单位)
        eps_inf: 高频介电常数
        poles: 极点列表 [(sigma, omega, gamma), ...]
               对于 Drude 项: omega ≈ 0
               对于 Lorentz 项: omega > 0
    
    Returns:
        复介电常数数组
    """
    omega = 2 * np.pi * freq
    epsilon = np.full_like(freq, eps_inf, dtype=complex)
    
    for sigma, omega0, gamma in poles:
        if omega0 < 1e-6:  # Drude 项
            epsilon += sigma / (-omega**2 - 1j * gamma * omega)
        else:  # Lorentz 项
            epsilon += sigma * omega0**2 / (omega0**2 - omega**2 - 1j * gamma * omega)
    
    return epsilon


def fit_lorentz_poles(wavelength_um, epsilon_data, num_poles=3, 
                      wavelength_range=None, include_drude=True):
    """
    从实验数据拟合 Lorentz-Drude 极点
    
    Args:
        wavelength_um: 波长数组 (μm)
        epsilon_data: 复介电常数数组
        num_poles: Lorentz 极点数量
        wavelength_range: 拟合的波长范围 (min, max)
        include_drude: 是否包含 Drude 项
    
    Returns:
        eps_inf: 高频介电常数
        poles: 极点列表
    """
    # 筛选数据范围
    if wavelength_range:
        mask = (wavelength_um >= wavelength_range[0]) & (wavelength_um <= wavelength_range[1])
        wl = wavelength_um[mask]
        eps = epsilon_data[mask]
    else:
        wl = wavelength_um
        eps = epsilon_data
    
    freq = 1.0 / wl
    
    def objective(params):
        """目标函数: 最小化拟合误差"""
        eps_inf = params[0]
        poles = []
        idx = 1
        
        if include_drude:
            sigma_d = params[idx]
            gamma_d = params[idx + 1]
            poles.append((sigma_d, 0, gamma_d))
            idx += 2
        
        for i in range(num_poles):
            sigma = params[idx]
            omega = params[idx + 1]
            gamma = params[idx + 2]
            poles.append((sigma, omega, gamma))
            idx += 3
        
        eps_fit = lorentz_drude_epsilon(freq, eps_inf, poles)
        
        # 计算相对误差
        error = np.sum(np.abs(eps_fit - eps)**2 / (np.abs(eps)**2 + 1))
        return error
    
    # 初始参数
    eps_inf_init = np.real(eps[-1])  # 高频极限
    
    bounds = [(0.5, 20)]  # eps_inf
    x0 = [eps_inf_init]
    
    if include_drude:
        bounds.extend([(0, 100), (0.001, 1)])  # sigma_d, gamma_d
        x0.extend([10, 0.1])
    
    for i in range(num_poles):
        bounds.extend([(0, 50), (0.01, 5), (0.001, 2)])  # sigma, omega, gamma
        x0.extend([1, 0.5 + i * 0.3, 0.1])
    
    # 优化
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    # 提取结果
    params = result.x
    eps_inf = params[0]
    poles = []
    idx = 1
    
    if include_drude:
        poles.append((params[idx], 0, params[idx + 1]))
        idx += 2
    
    for i in range(num_poles):
        poles.append((params[idx], params[idx + 1], params[idx + 2]))
        idx += 3
    
    return eps_inf, poles, result.fun


def create_meep_material_from_fit(eps_inf, poles):
    """
    从拟合的极点创建 MEEP 材料
    
    Args:
        eps_inf: 高频介电常数
        poles: 极点列表 [(sigma, omega, gamma), ...]
    
    Returns:
        mp.Medium 对象
    """
    susceptibilities = []
    
    for sigma, omega, gamma in poles:
        if omega < 1e-6:  # Drude 项
            susceptibilities.append(
                mp.DrudeSusceptibility(frequency=1.0, gamma=gamma, sigma=sigma)
            )
        else:  # Lorentz 项
            susceptibilities.append(
                mp.LorentzianSusceptibility(frequency=omega, gamma=gamma, 
                                            sigma=sigma)
            )
    
    return mp.Medium(epsilon=eps_inf, E_susceptibilities=susceptibilities)


def create_material_from_data(filepath, wavelength_range=(8, 15), num_poles=2):
    """
    从数据文件创建 MEEP 材料
    
    Args:
        filepath: 数据文件路径
        wavelength_range: 目标波长范围
        num_poles: Lorentz 极点数量
    
    Returns:
        mp.Medium 对象
    """
    wavelength_um, n, k = load_material_data(filepath)
    epsilon = nk_to_epsilon(n, k)
    
    # 检查是否为金属（负实部）
    mask = (wavelength_um >= wavelength_range[0]) & (wavelength_um <= wavelength_range[1])
    avg_eps_real = np.mean(np.real(epsilon[mask]))
    include_drude = avg_eps_real < 0
    
    # 拟合
    eps_inf, poles, error = fit_lorentz_poles(
        wavelength_um, epsilon, 
        num_poles=num_poles,
        wavelength_range=wavelength_range,
        include_drude=include_drude
    )
    
    print(f"Material fit from {filepath}:")
    print(f"  eps_inf = {eps_inf:.4f}")
    print(f"  Poles: {len(poles)}")
    for i, (s, w, g) in enumerate(poles):
        pole_type = "Drude" if w < 1e-6 else "Lorentz"
        print(f"    [{i}] {pole_type}: sigma={s:.4f}, omega={w:.4f}, gamma={g:.4f}")
    print(f"  Fit error: {error:.6f}")
    
    return create_meep_material_from_fit(eps_inf, poles)


def create_zns_material_fitted(filepath="ZnS.txt"):
    """
    使用多项式拟合创建 ZnS 材料
    ZnS 在远红外波段色散较弱，可以使用简单模型
    """
    wavelength_um, n, k = load_material_data(filepath)
    
    # 选择目标波长范围
    mask = (wavelength_um >= 8) & (wavelength_um <= 15)
    
    if np.sum(mask) < 3:
        # 数据不足，使用平均值
        n_avg = np.mean(n)
        k_avg = np.mean(k)
    else:
        # 使用多项式拟合
        wl_fit = wavelength_um[mask]
        n_fit = n[mask]
        k_fit = k[mask]
        
        # 二次多项式拟合
        n_coef = np.polyfit(wl_fit, n_fit, 2)
        k_coef = np.polyfit(wl_fit, k_fit, 2)
        
        # 在中心波长处的值
        wl_center = 12.5
        n_avg = np.polyval(n_coef, wl_center)
        k_avg = np.polyval(k_coef, wl_center)
    
    epsilon_real = n_avg**2 - k_avg**2
    epsilon_imag = 2 * n_avg * k_avg
    
    print(f"ZnS (polynomial fit): n = {n_avg:.4f}, k = {k_avg:.4f}")
    print(f"  epsilon = {epsilon_real:.4f} + {epsilon_imag:.4f}i")
    
    if abs(epsilon_imag) < 0.01:
        return mp.Medium(epsilon=epsilon_real)
    else:
        # 使用单个 Lorentz 极点模拟损耗
        # 在目标频率处的响应
        fcen = 1.0 / 12.5  # 中心频率
        
        return mp.Medium(epsilon=epsilon_real, D_conductivity=abs(epsilon_imag))


if __name__ == "__main__":
    print("Testing material interpolation and fitting...")
    print("=" * 60)
    
    # 测试 ZnS 拟合
    try:
        zns = create_zns_material_fitted("ZnS.txt")
        print(f"\nZnS material created successfully")
    except Exception as e:
        print(f"Error creating ZnS material: {e}")
    
    # 创建插值对象进行验证
    try:
        wl, n, k = load_material_data("ZnS.txt")
        interp_mat = InterpolatedMaterial(wl, n, k)
        
        print("\nInterpolated ZnS values:")
        for test_wl in [10, 12, 14]:
            n_val, k_val = interp_mat.get_nk(test_wl)
            eps = interp_mat.get_epsilon(test_wl)
            print(f"  λ = {test_wl} μm: n = {n_val:.4f}, k = {k_val:.4f}, ε = {eps:.4f}")
    except Exception as e:
        print(f"Error testing interpolation: {e}")
