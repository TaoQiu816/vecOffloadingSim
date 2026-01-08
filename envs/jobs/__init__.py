"""
[作业数据结构] envs/jobs/__init__.py
Job Data Structures for VEC Offloading Environment

作用 (Purpose):
    定义通信作业(TransferJob)和计算作业(ComputeJob)的数据结构。
    Defines data structures for communication jobs (TransferJob) and
    computation jobs (ComputeJob).

设计原则 (Design Principles):
    - 纯数据类，不包含业务逻辑
    - 支持时间戳跟踪和进度统计
    - 兼容FIFO队列系统
"""


class TransferJob:
    """
    [通信任务] 单个传输job（INPUT或EDGE）

    职责：
    - 记录传输来源/目标/剩余数据量
    - 区分INPUT（动作功率）与EDGE（固定最大功率）
    - 跟踪传输进度与时间戳

    Attributes:
        kind (str): "INPUT" 或 "EDGE"
        src_node (tuple): 源节点 ("VEH", i) 或 ("RSU", j)
        dst_node (tuple): 目标节点 ("RSU", j) 或 ("VEH", k)
        owner_vehicle_id (int): 该DAG属于哪辆车
        subtask_id (int): INPUT对应本subtask；EDGE对应child_id
        parent_task_id (int): EDGE必须有；INPUT为None
        rem_bytes (float): 剩余字节数
        tx_power_dbm (float): 发射功率（INPUT=动作映射；EDGE=MAX）
        link_type (str): "V2I" 或 "V2V"
        enqueue_time (float): 入队时间
        start_time (float): 开始传输时间
        finish_time (float): 完成时间
        step_time_used (float): 本step使用时间
        step_bytes_sent (float): 本step发送字节数
    """

    def __init__(self, kind, src_node, dst_node, owner_vehicle_id, subtask_id,
                 rem_bytes, tx_power_dbm, link_type, enqueue_time, parent_task_id=None):
        """
        初始化传输作业

        Args:
            kind: "INPUT" 或 "EDGE"
            src_node: ("VEH", i) 或 ("RSU", j)
            dst_node: ("RSU", j) 或 ("VEH", k)
            owner_vehicle_id: 该DAG属于哪辆车
            subtask_id: INPUT对应本subtask；EDGE对应child_id
            rem_bytes: 剩余字节数
            tx_power_dbm: 发射功率（INPUT=动作映射；EDGE=MAX）
            link_type: "V2I" 或 "V2V"
            enqueue_time: 入队时间
            parent_task_id: EDGE必须有；INPUT为None
        """
        self.kind = kind
        self.src_node = src_node
        self.dst_node = dst_node
        self.owner_vehicle_id = owner_vehicle_id
        self.subtask_id = subtask_id
        self.parent_task_id = parent_task_id
        self.rem_bytes = rem_bytes
        self.tx_power_dbm = tx_power_dbm
        self.link_type = link_type

        # 时间戳
        self.enqueue_time = enqueue_time
        self.start_time = None
        self.finish_time = None

        # 本step统计
        self.step_time_used = 0.0
        self.step_bytes_sent = 0.0

    def get_unique_key(self):
        """
        获取唯一键（用于EDGE去重）

        Returns:
            tuple: (owner_vehicle_id, subtask_id, parent_task_id) 或
                   (owner_vehicle_id, subtask_id, "INPUT")
        """
        if self.kind == "EDGE":
            return (self.owner_vehicle_id, self.subtask_id, self.parent_task_id)
        else:
            return (self.owner_vehicle_id, self.subtask_id, "INPUT")


class ComputeJob:
    """
    [计算任务] 单个计算job

    职责：
    - 记录计算位置（车辆或RSU的处理器ID）
    - 跟踪剩余计算量与进度
    - 支持多处理器并行（RSU）

    Attributes:
        owner_vehicle_id (int): 任务所属车辆ID
        subtask_id (int): 子任务ID
        rem_cycles (float): 剩余计算量（cycles）
        exec_node (tuple): ("VEH", i) 或 ("RSU", j)
        processor_id (int): 车辆恒0；RSU为[0..P-1]
        enqueue_time (float): 入队时间
        start_time (float): 开始计算时间
        finish_time (float): 完成时间
        step_time_used (float): 本step使用时间
        step_cycles_done (float): 本step完成的计算量
    """

    def __init__(self, owner_vehicle_id, subtask_id, rem_cycles, exec_node,
                 processor_id, enqueue_time):
        """
        初始化计算作业

        Args:
            owner_vehicle_id: 任务所属车辆ID
            subtask_id: 子任务ID
            rem_cycles: 剩余计算量（cycles）
            exec_node: ("VEH", i) 或 ("RSU", j)
            processor_id: 车辆恒0；RSU为[0..P-1]
            enqueue_time: 入队时间
        """
        self.owner_vehicle_id = owner_vehicle_id
        self.subtask_id = subtask_id
        self.rem_cycles = rem_cycles
        self.exec_node = exec_node
        self.processor_id = processor_id

        # 时间戳
        self.enqueue_time = enqueue_time
        self.start_time = None
        self.finish_time = None

        # 本step统计
        self.step_time_used = 0.0
        self.step_cycles_done = 0.0


# 导出列表
__all__ = ['TransferJob', 'ComputeJob']
