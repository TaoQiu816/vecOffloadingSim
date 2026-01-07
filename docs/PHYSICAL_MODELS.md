# 物理模型规范 (Physical Models Specification)

本文档详细记录仿真系统中的通信、计算、移动性等物理模型的数学公式和参数配置。

---

## 1. 通信模型（C-V2X）

### 1.1 频谱资源分配

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| V2I带宽 | $B_{V2I}$ | 20 MHz | 车辆到RSU上行链路 |
| V2V带宽 | $B_{V2V}$ | 10 MHz | 车辆间直接通信 |
| 噪声功率谱密度 | $N_0$ | -174 dBm/Hz | 热噪声 |

### 1.2 传输速率计算（Shannon公式）

$$
R = B \cdot \log_2\left(1 + \frac{P_{rx}}{N_0 \cdot B + I}\right)
$$

其中：
- $R$：传输速率（bps）
- $B$：带宽（Hz）
- $P_{rx}$：接收功率（W）
- $N_0$：噪声功率谱密度（W/Hz）
- $I$：干扰功率（W）

### 1.3 路径损耗模型

采用对数距离路径损耗模型：

$$
PL(d) = PL_0 + 10 \cdot n \cdot \log_{10}\left(\frac{d}{d_0}\right) + X_\sigma
$$

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 参考距离 | $d_0$ | 1 m | 参考点距离 |
| 参考损耗 | $PL_0$ | 38 dB | 参考点损耗（V2V） |
| 路径损耗指数 | $n$ | 2.7 | V2V典型值 |
| 阴影衰落 | $X_\sigma$ | $\mathcal{N}(0, 3^2)$ dB | 对数正态阴影 |

**V2I路径损耗**（城市宏蜂窝）：
$$
PL_{V2I}(d) = 128.1 + 37.6 \cdot \log_{10}(d) \quad [\text{dB}]
$$

**V2V路径损耗**（LOS场景）：
$$
PL_{V2V}(d) = 38 + 27 \cdot \log_{10}(d) \quad [\text{dB}]
$$

### 1.4 瑞利衰落

小尺度衰落建模为瑞利分布：

$$
h \sim \mathcal{CN}(0, 1)
$$

$$
|h|^2 \sim \text{Exp}(1)
$$

接收功率计算：
$$
P_{rx} = P_{tx} \cdot |h|^2 \cdot 10^{-PL/10}
$$

### 1.5 发射功率配置

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 最小发射功率 | $P_{tx,min}$ | 10 dBm | 最低传输功率 |
| 最大发射功率 | $P_{tx,max}$ | 23 dBm | 最高传输功率 |
| 电路功率 | $P_{circuit}$ | 100 mW | 基带处理功耗 |

动作输出的`power`为$[0,1]$范围，映射公式：
$$
P_{tx} = P_{tx,min} + power \cdot (P_{tx,max} - P_{tx,min}) \quad [\text{dBm}]
$$

### 1.6 通信范围

| 参数 | 默认值 | 说明 |
|------|--------|------|
| RSU覆盖半径 | 400 m | 车辆进入该范围可与RSU通信 |
| V2V通信范围 | 300 m | 车辆间距小于此值可建立V2V链路 |

---

## 2. 计算模型

### 2.1 车辆计算资源

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 最小CPU频率 | $f_{veh,min}$ | 1.0 GHz | 车载处理器下限 |
| 最大CPU频率 | $f_{veh,max}$ | 3.0 GHz | 车载处理器上限 |
| 功耗系数 | $\kappa$ | $10^{-27}$ | CMOS功耗系数 |

车辆CPU频率在初始化时从均匀分布采样：
$$
f_{veh} \sim \mathcal{U}(f_{veh,min}, f_{veh,max})
$$

### 2.2 RSU计算资源

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| RSU CPU频率 | $f_{RSU}$ | 12.0 GHz | 边缘服务器主频 |
| 处理器核数 | $N_{proc}$ | 4 | 并行处理能力 |

RSU采用多核并行处理，负载均衡分配到各处理器队列。

### 2.3 任务执行时间

$$
T_{exec} = \frac{C}{f}
$$

其中：
- $T_{exec}$：执行时间（秒）
- $C$：计算量（CPU cycles）
- $f$：处理器频率（Hz）

### 2.4 动态功耗模型

CPU动态功耗：
$$
P_{cpu} = \kappa \cdot f^3
$$

单任务执行能耗：
$$
E_{exec} = P_{cpu} \cdot T_{exec} = \kappa \cdot f^2 \cdot C
$$

---

## 3. DAG任务模型

### 3.1 DAG生成参数

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 最小节点数 | $N_{min}$ | 5 | 子任务最少数量 |
| 最大节点数 | $N_{max}$ | 10 | 子任务最多数量 |
| 最小计算量 | $C_{min}$ | 0.8 GCycles | 单任务最小工作量 |
| 最大计算量 | $C_{max}$ | 2.5 GCycles | 单任务最大工作量 |
| 最小数据量 | $D_{min}$ | 1.0 Mbits | 单任务最小输入数据 |
| 最大数据量 | $D_{max}$ | 4.0 Mbits | 单任务最大输入数据 |

### 3.2 DAG结构生成

采用分层随机DAG生成算法：

1. **层数确定**：$L \sim \mathcal{U}(2, 4)$
2. **每层节点数**：按节点总数均匀分配到各层
3. **边生成**：相邻层间随机连接，确保连通性
4. **数据量分配**：边数据量 $d_{ij} \sim \mathcal{U}(D_{min}, D_{max})$

### 3.3 Deadline计算

$$
T_{deadline} = \gamma \cdot T_{critical}
$$

其中：
- $\gamma$：紧缩系数，$\gamma \sim \mathcal{U}(0.70, 0.80)$
- $T_{critical}$：关键路径执行时间（假设全本地执行）

关键路径计算：
$$
T_{critical} = \max_{path \in DAG} \sum_{i \in path} \frac{C_i}{f_{veh,max}}
$$

### 3.4 子任务状态机

```
PENDING (0) ──────► READY (1) ──────► EXECUTING (2) ──────► COMPLETED (3)
                        │                   │
                        └───────────────────┴──────────► FAILED (4)
```

| 状态 | ID | 转换条件 |
|------|------|----------|
| PENDING | 0 | 初始状态，等待前驱完成 |
| READY | 1 | 所有前驱已完成，可被调度 |
| EXECUTING | 2 | 已提交执行决策，正在传输/计算 |
| COMPLETED | 3 | 执行完成（含结果传输） |
| FAILED | 4 | 超时或执行失败 |

---

## 4. 移动性模型

### 4.1 道路场景

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 道路长度 | 2000 m | 单向道路 |
| 车道数 | 4 | 单向4车道 |
| 车道宽度 | 3.5 m | 标准车道 |
| RSU位置 | (1000, 0) | 道路中点 |

### 4.2 车辆到达模型

采用泊松过程：
$$
N(t) \sim \text{Poisson}(\lambda \cdot t)
$$

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 到达率 | $\lambda$ | 0.5 veh/s | 每秒平均到达车辆数 |

### 4.3 速度分布

采用截断正态分布：
$$
v \sim \text{TruncatedNormal}(\mu_v, \sigma_v, v_{min}, v_{max})
$$

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 平均速度 | $\mu_v$ | 13.8 m/s | 约50 km/h |
| 速度标准差 | $\sigma_v$ | 3.0 m/s | 速度波动 |
| 最小速度 | $v_{min}$ | 5.0 m/s | 18 km/h |
| 最大速度 | $v_{max}$ | 20.0 m/s | 72 km/h |

### 4.4 位置更新

简化为匀速直线运动：
$$
x(t + \Delta t) = x(t) + v_x \cdot \Delta t
$$

$$
y(t + \Delta t) = y(t) + v_y \cdot \Delta t
$$

对于单向道路，$v_y = 0$。

### 4.5 接触时间估算

V2V接触时间（两车在通信范围内的剩余时间）：

设车辆A位置$(x_A, y_A)$，速度$v_A$；车辆B位置$(x_B, y_B)$，速度$v_B$。

相对速度：$v_{rel} = v_A - v_B$

相对距离：$d = \sqrt{(x_A - x_B)^2 + (y_A - y_B)^2}$

接触时间估算（同向行驶）：
$$
T_{contact} = \frac{R_{V2V} - d}{|v_{rel}|} \quad \text{if } v_{rel} \cdot (x_A - x_B) < 0
$$

---

## 5. 能耗模型

### 5.1 传输能耗

$$
E_{tx} = P_{tx} \cdot T_{tx} + P_{circuit} \cdot T_{tx}
$$

其中：
- $P_{tx}$：发射功率（W）
- $T_{tx}$：传输时间（s）
- $P_{circuit}$：电路功耗（W）

### 5.2 计算能耗

$$
E_{comp} = \kappa \cdot f^2 \cdot C
$$

### 5.3 总能耗

$$
E_{total} = E_{tx} + E_{comp}
$$

### 5.4 能耗归一化

为了RL训练稳定，对能耗进行归一化：
$$
E_{norm} = \frac{E_{total}}{E_{max}}
$$

其中 $E_{max}$ 为预设的最大能耗参考值。

---

## 6. 时间参数

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 仿真步长 | $\Delta t$ (DT) | 10 ms | 每step推进时间 |
| 最大步数 | MAX_STEPS | 500 | 单episode最大步数 |
| 仿真时长 | $T_{max}$ | 5 s | = DT × MAX_STEPS |

---

## 7. 配置文件对照

所有物理参数定义在 `configs/config.py` 的 `SystemConfig` 类中：

```python
class SystemConfig:
    # 通信参数
    BW_V2I = 20e6           # V2I带宽 (Hz)
    BW_V2V = 10e6           # V2V带宽 (Hz)
    V2V_RANGE = 300.0       # V2V通信范围 (m)
    RSU_RANGE = 400.0       # RSU覆盖范围 (m)
    TX_POWER_MIN_DBM = 10   # 最小发射功率 (dBm)
    TX_POWER_MAX_DBM = 23   # 最大发射功率 (dBm)

    # 计算参数
    MIN_VEHICLE_CPU_FREQ = 1.0e9   # 最小车辆CPU (Hz)
    MAX_VEHICLE_CPU_FREQ = 3.0e9   # 最大车辆CPU (Hz)
    F_RSU = 12.0e9                  # RSU CPU频率 (Hz)
    RSU_NUM_PROCESSORS = 4          # RSU处理器数

    # DAG任务参数
    MIN_NODES = 5
    MAX_NODES = 10
    MIN_COMP = 0.8e9        # 最小计算量 (cycles)
    MAX_COMP = 2.5e9        # 最大计算量 (cycles)
    MIN_DATA = 1.0e6        # 最小数据量 (bits)
    MAX_DATA = 4.0e6        # 最大数据量 (bits)
    DEADLINE_TIGHTENING_MIN = 0.70
    DEADLINE_TIGHTENING_MAX = 0.80

    # 移动性参数
    VEL_MEAN = 13.8         # 平均速度 (m/s)
    VEL_STD = 3.0           # 速度标准差 (m/s)
    VEL_MIN = 5.0           # 最小速度 (m/s)
    VEL_MAX = 20.0          # 最大速度 (m/s)
    VEHICLE_ARRIVAL_RATE = 0.5  # 到达率 (veh/s)

    # 时间参数
    DT = 0.01               # 仿真步长 (s)
    MAX_STEPS = 500         # 最大步数
```

---

## 8. 公式符号表

| 符号 | 含义 | 单位 |
|------|------|------|
| $R$ | 传输速率 | bps |
| $B$ | 带宽 | Hz |
| $P_{tx}$ | 发射功率 | W 或 dBm |
| $P_{rx}$ | 接收功率 | W |
| $N_0$ | 噪声功率谱密度 | W/Hz |
| $PL$ | 路径损耗 | dB |
| $h$ | 信道衰落系数 | - |
| $f$ | 处理器频率 | Hz |
| $C$ | 计算量 | cycles |
| $D$ | 数据量 | bits |
| $T$ | 时间 | s |
| $E$ | 能耗 | J |
| $v$ | 速度 | m/s |
| $d$ | 距离 | m |
| $\gamma$ | Deadline紧缩系数 | - |
| $\lambda$ | 车辆到达率 | veh/s |
| $\kappa$ | CPU功耗系数 | - |
