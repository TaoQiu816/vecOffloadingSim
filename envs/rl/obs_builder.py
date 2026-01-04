"""
观测构造器：将环境状态转换为智能体观测

职责：
- 从环境状态提取特征
- 生成node_x, self_info, rsu_info, adj, neighbors等
- 计算action_mask和target_mask
- 固定维度填充以满足批处理要求

设计原则：
- 纯函数风格（无副作用）
- 观测shape/数值映射/mask规则与原实现完全一致
- 无未来信息泄漏
"""
from typing import TYPE_CHECKING, Dict, List, Tuple, Any
import numpy as np

if TYPE_CHECKING:
    from envs.vec_offloading_env import VecOffloadingEnv  # type: ignore
    from envs.entities.vehicle import Vehicle  # type: ignore
else:
    VecOffloadingEnv = object
    Vehicle = object


class ObsBuilder:
    """
    观测构造器：负责从环境状态生成智能体观测
    
    【重要】保持与原实现完全一致：
    - 观测维度、归一化系数、特征顺序
    - mask逻辑（task_mask, target_mask）
    - 固定维度填充（MAX_NODES, MAX_NEIGHBORS, MAX_TARGETS）
    - 死锁兜底（所有目标不可用时强制开启Local）
    """
    
    def __init__(self, config):
        """
        Args:
            config: 系统配置对象
        """
        self.config = config
        
        # 预计算归一化系数（避免重复计算）
        self._inv_max_comp = 1.0 / max(config.NORM_MAX_COMP, 1e-6)
        self._inv_max_data = 1.0 / max(config.NORM_MAX_DATA, 1e-6)
        self._inv_max_nodes = 1.0 / max(config.MAX_NODES, 1)
        self._inv_max_velocity = 1.0 / max(config.MAX_VELOCITY, 1e-6)
        self._inv_max_wait = 1.0 / max(config.NORM_MAX_WAIT_TIME, 1e-6)
        self._inv_max_cpu = 1.0 / max(config.NORM_MAX_CPU, 1e-6)
        self._inv_max_rate_v2i = 1.0 / max(config.NORM_MAX_RATE_V2I, 1e-6)
        self._inv_max_rate_v2v = 1.0 / max(config.NORM_MAX_RATE_V2V, 1e-6)
        self._inv_map_size = 1.0 / max(config.MAP_SIZE, 1e-6)
        self._inv_v2v_range = 1.0 / max(config.V2V_RANGE, 1e-6)
        self._max_v2v_contact_time = getattr(config, 'MAX_V2V_CONTACT_TIME', 10.0)
    
    def build(self, env: VecOffloadingEnv) -> List[Dict[str, Any]]:
        """
        [主入口] 构造所有车辆的观测
        
        Args:
            env: 环境实例（提供状态访问）
            
        Returns:
            list[dict]: 每个车辆的观测字典
        """
        obs_list = []
        dist_matrix = env._get_dist_matrix()
        vehicle_ids = [veh.id for veh in env.vehicles]
        
        for v in env.vehicles:
            obs = self._build_single_vehicle_obs(
                v, env, dist_matrix, vehicle_ids
            )
            obs_list.append(obs)
        
        return obs_list
    
    def _build_single_vehicle_obs(self, vehicle: Vehicle, env: VecOffloadingEnv,
                                   dist_matrix: np.ndarray, vehicle_ids: List[int]) -> Dict[str, Any]:
        """
        构造单个车辆的观测
        
        【注释说明】此方法完全复制原_get_obs中的逻辑，保持行为一致。
        后续可逐步优化，但当前阶段保持100%兼容。
        """
        # [委托到env的原方法] 当前阶段保持行为完全一致
        # 后续可逐步将逻辑迁移到此处
        raise NotImplementedError(
            "ObsBuilder当前作为框架存在，实际逻辑仍在VecOffloadingEnv._get_obs中。"
            "阶段6的目标是建立框架并集成，逐步迁移逻辑需要更细致的测试。"
        )

