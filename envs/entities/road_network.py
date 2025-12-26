"""
道路网络模型

实现道路网络，支持：
- 道路段（Road Segment）
- 车辆在道路上的移动约束
- RSU在道路上的部署
- 道路拓扑结构（可选：支持交叉口）
"""

import numpy as np
from configs.config import SystemConfig as Cfg


class RoadSegment:
    """
    道路段实体
    
    表示一条道路段，车辆可以在道路上移动
    """
    
    def __init__(self, segment_id, start_point, end_point, width=None):
        """
        初始化道路段
        
        Args:
            segment_id: 道路段ID
            start_point: 起点坐标 [x, y]
            end_point: 终点坐标 [x, y]
            width: 道路宽度 (m)，如果为None则使用默认值
        """
        self.id = segment_id
        self.start = np.array(start_point, dtype=float)
        self.end = np.array(end_point, dtype=float)
        self.width = width if width is not None else 20.0  # 默认道路宽度20m
        
        # 计算道路方向和长度
        self.direction = self.end - self.start
        self.length = np.linalg.norm(self.direction)
        if self.length > 0:
            self.direction = self.direction / self.length  # 归一化方向向量
        else:
            self.direction = np.array([1.0, 0.0])  # 默认方向
        
        # 垂直方向（用于计算道路边界）
        self.perpendicular = np.array([-self.direction[1], self.direction[0]])
    
    def get_position_at_distance(self, distance):
        """
        获取道路上指定距离处的坐标
        
        Args:
            distance: 距离起点的距离 (m)，0表示起点，length表示终点
        
        Returns:
            np.array: 坐标 [x, y]
        """
        distance = np.clip(distance, 0, self.length)
        return self.start + self.direction * distance
    
    def project_point_to_road(self, point):
        """
        将点投影到道路上（找到道路上最近的点）
        
        Args:
            point: 点坐标 [x, y]
        
        Returns:
            tuple: (投影点坐标, 距离起点的距离, 到道路的距离)
        """
        point = np.array(point)
        vec_to_start = point - self.start
        
        # 投影到道路方向
        proj_length = np.dot(vec_to_start, self.direction)
        proj_length = np.clip(proj_length, 0, self.length)
        
        # 投影点
        proj_point = self.start + self.direction * proj_length
        
        # 到道路的距离（垂直距离）
        dist_to_road = np.linalg.norm(point - proj_point)
        
        return proj_point, proj_length, dist_to_road
    
    def is_point_on_road(self, point, tolerance=None):
        """
        检查点是否在道路上（考虑道路宽度）
        
        Args:
            point: 点坐标 [x, y]
            tolerance: 容差 (m)，如果为None则使用道路宽度的一半
        
        Returns:
            bool: 如果点在道路上返回True
        """
        if tolerance is None:
            tolerance = self.width / 2
        
        _, _, dist_to_road = self.project_point_to_road(point)
        return dist_to_road <= tolerance
    
    def get_bounds(self):
        """
        获取道路段的边界框
        
        Returns:
            tuple: (min_x, min_y, max_x, max_y)
        """
        corners = [
            self.start + self.perpendicular * (self.width / 2),
            self.start - self.perpendicular * (self.width / 2),
            self.end + self.perpendicular * (self.width / 2),
            self.end - self.perpendicular * (self.width / 2)
        ]
        corners = np.array(corners)
        return (corners[:, 0].min(), corners[:, 1].min(), 
                corners[:, 0].max(), corners[:, 1].max())


class RoadNetwork:
    """
    道路网络
    
    管理多个道路段，提供道路查询和车辆约束
    """
    
    def __init__(self, map_size=None):
        """
        初始化道路网络
        
        Args:
            map_size: 地图大小，如果为None则使用配置值
        """
        self.map_size = map_size if map_size is not None else Cfg.MAP_SIZE
        self.segments = []
        
        # 默认创建一个简单的网格道路网络（交叉道路）
        self._create_default_road_network()
    
    def _create_default_road_network(self):
        """
        创建默认的道路网络（网格状）
        
        创建水平和垂直的主要道路
        """
        size = self.map_size
        margin = size * 0.1  # 边距
        
        # 水平道路（3条）
        for i, y in enumerate([size * 0.25, size * 0.5, size * 0.75]):
            seg = RoadSegment(
                segment_id=len(self.segments),
                start_point=[margin, y],
                end_point=[size - margin, y],
                width=20.0
            )
            self.segments.append(seg)
        
        # 垂直道路（3条）
        for i, x in enumerate([size * 0.25, size * 0.5, size * 0.75]):
            seg = RoadSegment(
                segment_id=len(self.segments),
                start_point=[x, margin],
                end_point=[x, size - margin],
                width=20.0
            )
            self.segments.append(seg)
    
    def add_segment(self, start_point, end_point, width=None):
        """
        添加道路段
        
        Args:
            start_point: 起点坐标
            end_point: 终点坐标
            width: 道路宽度
        
        Returns:
            RoadSegment: 创建的道路段
        """
        seg = RoadSegment(
            segment_id=len(self.segments),
            start_point=start_point,
            end_point=end_point,
            width=width
        )
        self.segments.append(seg)
        return seg
    
    def find_nearest_road(self, position):
        """
        找到距离指定位置最近的道路段
        
        Args:
            position: 位置坐标 [x, y]
        
        Returns:
            tuple: (最近的道路段, 投影点, 距离起点的距离, 到道路的距离)
        """
        if len(self.segments) == 0:
            return None, None, 0, float('inf')
        
        min_dist = float('inf')
        nearest_seg = None
        nearest_proj = None
        nearest_dist_along = 0
        
        for seg in self.segments:
            proj_point, dist_along, dist_to_road = seg.project_point_to_road(position)
            if dist_to_road < min_dist:
                min_dist = dist_to_road
                nearest_seg = seg
                nearest_proj = proj_point
                nearest_dist_along = dist_along
        
        return nearest_seg, nearest_proj, nearest_dist_along, min_dist
    
    def constrain_position_to_road(self, position, max_offset=None):
        """
        将位置约束到道路上（投影到最近的道路）
        
        Args:
            position: 位置坐标 [x, y]
            max_offset: 最大偏移距离，如果为None则使用道路宽度
        
        Returns:
            np.array: 约束后的位置
        """
        seg, proj_point, _, dist_to_road = self.find_nearest_road(position)
        
        if seg is None:
            return np.array(position)
        
        # 如果距离道路太远，则约束到道路上
        if max_offset is None:
            max_offset = seg.width / 2
        
        if dist_to_road > max_offset:
            return proj_point
        
        return np.array(position)
    
    def get_road_direction_at(self, position):
        """
        获取指定位置处道路的方向
        
        Args:
            position: 位置坐标 [x, y]
        
        Returns:
            np.array: 道路方向向量（归一化）
        """
        seg, _, _, _ = self.find_nearest_road(position)
        if seg is None:
            return np.array([1.0, 0.0])  # 默认方向
        return seg.direction
    
    def is_position_on_road(self, position):
        """
        检查位置是否在道路上
        
        Args:
            position: 位置坐标 [x, y]
        
        Returns:
            bool: 如果在道路上返回True
        """
        seg, _, _, dist_to_road = self.find_nearest_road(position)
        if seg is None:
            return False
        return dist_to_road <= seg.width / 2
    
    def get_road_segments_in_range(self, position, radius):
        """
        获取指定范围内的道路段
        
        Args:
            position: 中心位置 [x, y]
            radius: 范围半径 (m)
        
        Returns:
            list: 范围内的道路段列表
        """
        result = []
        position = np.array(position)
        
        for seg in self.segments:
            # 检查道路段的起点、终点和中点
            points = [seg.start, seg.end, (seg.start + seg.end) / 2]
            for point in points:
                if np.linalg.norm(position - point) <= radius:
                    result.append(seg)
                    break
        
        return result

