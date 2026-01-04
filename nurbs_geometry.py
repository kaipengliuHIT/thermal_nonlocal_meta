"""
NURBS B样条曲线几何生成模块
将控制点转换为 MEEP 可用的多边形几何
"""

import numpy as np
import meep as mp


def basis_function(i, t):
    """二次B样条基函数"""
    if i == 1:
        return (1 - t) * (1 - t)
    if i == 2:
        return 2 * t * (1 - t)
    if i == 3:
        return t * t
    return 0


def generate_nurbs_curve(control_points, num_points_per_segment=100):
    """
    根据控制点生成 NURBS 曲线
    
    Args:
        control_points: 控制点数组，形状 (N, 2)
        num_points_per_segment: 每段曲线的采样点数
    
    Returns:
        center_points: 曲线上的点坐标，形状 (M, 2)
    """
    points = np.array(control_points)
    segments = len(points) - 2
    
    if segments < 1:
        return None
    
    total_points = segments * num_points_per_segment
    center_points = np.zeros((total_points, 2))
    
    for seg in range(segments):
        for j in range(num_points_per_segment):
            t = j / (num_points_per_segment - 1)
            
            x = (basis_function(1, t) * points[seg][0] +
                 basis_function(2, t) * points[seg + 1][0] +
                 basis_function(3, t) * points[seg + 2][0])
            
            y = (basis_function(1, t) * points[seg][1] +
                 basis_function(2, t) * points[seg + 1][1] +
                 basis_function(3, t) * points[seg + 2][1])
            
            index = seg * num_points_per_segment + j
            center_points[index] = [x, y]
    
    return center_points


def curve_to_polygon_vertices(center_points, line_width):
    """
    将曲线中心点转换为多边形顶点（考虑线宽）
    
    Args:
        center_points: 曲线中心点
        line_width: 线宽
    
    Returns:
        vertices: 多边形顶点数组
    """
    # 计算切线方向
    tangents = np.zeros_like(center_points)
    tangents[1:-1] = center_points[2:] - center_points[:-2]
    tangents[0] = center_points[1] - center_points[0]
    tangents[-1] = center_points[-1] - center_points[-2]
    
    # 归一化切线向量
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms == 0] = 1e-10
    tangents /= norms[:, np.newaxis]
    
    # 计算法线向量（旋转90度）
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    
    # 计算多边形顶点
    top_points = center_points + normals * line_width / 2
    bottom_points = center_points - normals * line_width / 2
    
    # 创建闭合多边形
    vertices = np.vstack((
        top_points,
        bottom_points[::-1]
    ))
    
    return vertices


def create_nurbs_prism(control_points, line_width, z_center, z_height, material):
    """
    创建 NURBS 曲线形状的棱柱（用于 MEEP）
    
    Args:
        control_points: 控制点数组
        line_width: 线宽 (um)
        z_center: z 方向中心位置 (um)
        z_height: z 方向高度 (um)
        material: MEEP 材料
    
    Returns:
        mp.Prism 对象
    """
    # 生成曲线点
    center_points = generate_nurbs_curve(control_points, num_points_per_segment=50)
    
    if center_points is None:
        return None
    
    # 生成多边形顶点
    vertices = curve_to_polygon_vertices(center_points, line_width)
    
    # 转换为 MEEP Vector3 顶点列表
    meep_vertices = [mp.Vector3(v[0], v[1]) for v in vertices]
    
    # 创建棱柱
    prism = mp.Prism(
        vertices=meep_vertices,
        height=z_height,
        center=mp.Vector3(0, 0, z_center),
        material=material
    )
    
    return prism


def create_nurbs_array(control_points_list, line_widths, z_center, z_height, 
                       material, period, array_size=5):
    """
    创建 NURBS 结构的周期性阵列
    
    Args:
        control_points_list: 控制点列表，每个元素是一组控制点
        line_widths: 每条曲线的线宽列表 (um)
        z_center: z 方向中心位置 (um)
        z_height: z 方向高度 (um)
        material: MEEP 材料
        period: 阵列周期 (um)
        array_size: 阵列大小 (NxN)
    
    Returns:
        geometry_list: MEEP 几何对象列表
    """
    geometry_list = []
    
    for idx, (control_points, line_width) in enumerate(zip(control_points_list, line_widths)):
        # 生成曲线点
        center_points = generate_nurbs_curve(control_points, num_points_per_segment=30)
        
        if center_points is None:
            continue
        
        # 生成多边形顶点
        vertices = curve_to_polygon_vertices(center_points, line_width)
        
        # 创建阵列
        for row in range(array_size):
            for col in range(array_size):
                # 计算偏移量
                x_offset = (col - array_size // 2) * period
                y_offset = (row - array_size // 2) * period
                
                # 应用偏移
                offset_vertices = vertices.copy()
                offset_vertices[:, 0] += x_offset
                offset_vertices[:, 1] += y_offset
                
                # 转换为 MEEP 顶点
                meep_vertices = [mp.Vector3(v[0], v[1]) for v in offset_vertices]
                
                # 创建棱柱
                prism = mp.Prism(
                    vertices=meep_vertices,
                    height=z_height,
                    center=mp.Vector3(0, 0, z_center),
                    material=material
                )
                
                geometry_list.append(prism)
    
    return geometry_list


def create_single_period_nurbs(control_points_list, line_widths, z_center, z_height, material):
    """
    创建单周期内的 NURBS 结构（用于周期边界条件）
    
    Args:
        control_points_list: 控制点列表
        line_widths: 线宽列表 (um)
        z_center: z 方向中心
        z_height: z 方向高度
        material: MEEP 材料
    
    Returns:
        geometry_list: MEEP 几何对象列表
    """
    geometry_list = []
    
    for control_points, line_width in zip(control_points_list, line_widths):
        center_points = generate_nurbs_curve(control_points, num_points_per_segment=30)
        
        if center_points is None:
            continue
        
        vertices = curve_to_polygon_vertices(center_points, line_width)
        meep_vertices = [mp.Vector3(v[0], v[1]) for v in vertices]
        
        prism = mp.Prism(
            vertices=meep_vertices,
            height=z_height,
            center=mp.Vector3(0, 0, z_center),
            material=material
        )
        
        geometry_list.append(prism)
    
    return geometry_list


if __name__ == "__main__":
    # 测试几何生成
    print("Testing NURBS geometry generation...")
    
    # 测试控制点
    test_points = np.array([[-5, -4], [-5.25, 0], [-5, 4]])
    
    # 生成曲线
    curve = generate_nurbs_curve(test_points, num_points_per_segment=50)
    print(f"Generated curve with {len(curve)} points")
    
    # 生成多边形顶点
    vertices = curve_to_polygon_vertices(curve, line_width=1.0)
    print(f"Generated polygon with {len(vertices)} vertices")
