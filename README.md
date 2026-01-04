# 热辐射非局域超表面仿真 (MEEP 并行版本)

基于 MEEP 的热辐射超表面电磁仿真，支持 MPI 并行计算。

## 项目结构

```
thermal_nonlocal_meta/
├── materials.py          # 材料定义模块 (ZnS, Ge, Au, Ag)
├── nurbs_geometry.py     # NURBS B样条曲线几何生成
├── thermal_meta_sim.py   # 主仿真脚本
├── run_sweep.py          # 参数扫描脚本
├── ZnS.txt               # ZnS 材料色散数据
└── README.md
```

## 结构说明

仿真的超表面结构从上到下：
- **Ag 纳米线阵列** (50 nm) - NURBS B样条曲线形状
- **Ge 顶层** (100 nm)
- **ZnS 中间层** (600 nm)
- **Ge 底层** (100 nm)
- **Au 反射层** (50 nm)

### 仿真参数
- 波长范围: 11.6 - 13.6 μm
- 周期: 12.6 μm
- 边界条件: x, y 方向周期边界, z 方向 PML

## 环境要求

```bash
# 激活 MEEP 环境
conda activate meep

# 验证安装
python -c "import meep as mp; print(mp.__version__)"
```

## 使用方法

### 单进程运行

```bash
conda activate meep
python thermal_meta_sim.py --resolution 10 --until 500 --output results
```

### 并行运行 (推荐)

```bash
conda activate meep
mpirun -np 4 python thermal_meta_sim.py --resolution 10 --until 500 --output results
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--resolution` | 网格分辨率 (pixels/um) | 10 |
| `--until` | 仿真时长 | 500 |
| `--output` | 输出文件名前缀 | results |
| `--simple` | 简化模式（不做归一化） | False |

### 参数扫描

```bash
# 扫描线宽
mpirun -np 4 python run_sweep.py --sweep-type width --resolution 8

# 扫描周期
mpirun -np 4 python run_sweep.py --sweep-type period --resolution 8

# 扫描曲率
mpirun -np 4 python run_sweep.py --sweep-type curvature --resolution 8
```

## 输出文件

- `results.csv` - 光谱数据 (波长, 反射率, 透射率)
- `results.png` - 光谱图
- `sweep_results.json` - 参数扫描结果

## 自定义仿真

修改 `thermal_meta_sim.py` 中的控制点定义:

```python
# NURBS 控制点定义
control_points = [
    [[x1, y1], [x2, y2], [x3, y3]],  # 第一条曲线
    [[x4, y4], [x5, y5], [x6, y6]],  # 第二条曲线
    ...
]

# 对应的线宽
line_widths = [1.1, 1.0, 0.8, ...]
```

## 与 Lumerical 版本的对应关系

| Lumerical | MEEP |
|-----------|------|
| `addrect` | `mp.Block` |
| `addpoly` | `mp.Prism` |
| `addplane` | `mp.Source` |
| `addpower` | `mp.FluxRegion` |
| Periodic BC | `k_point=mp.Vector3(0,0,0)` |

## 注意事项

1. 分辨率设置影响计算精度和速度，建议初始测试用 8-10，正式计算用 15-20
2. 金属材料在远红外波段的 Drude 模型参数可能需要根据实际情况调整
3. 并行计算时进程数不要超过 CPU 核心数
