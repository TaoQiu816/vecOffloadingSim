import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_full_system_lifecycle():
    # 备份并设置最小规模配置
    orig = {
        "NUM_VEHICLES": Cfg.NUM_VEHICLES,
        "NUM_RSU": Cfg.NUM_RSU,
        "VEHICLE_ARRIVAL_RATE": Cfg.VEHICLE_ARRIVAL_RATE,
        "MAX_STEPS": Cfg.MAX_STEPS,
    }
    Cfg.NUM_VEHICLES = 2
    Cfg.NUM_RSU = 1
    Cfg.VEHICLE_ARRIVAL_RATE = 0.0  # 保持车辆集合稳定
    Cfg.MAX_STEPS = 50
    try:
        env = VecOffloadingEnv()
        obs_list, _ = env.reset(seed=0)

        v0_id = env.vehicles[0].id
        v1_id = env.vehicles[1].id
        sub0 = int(obs_list[0]["subtask_index"])
        sub1 = int(obs_list[1]["subtask_index"])

        # 强制 Vehicle0 -> RSU0，Vehicle1 -> Vehicle0 (首个邻居)
        assert obs_list[0]["action_mask"][1], "Vehicle0 无法选择 RSU"
        rsu_target_idx = 1
        neighbor_indices = np.where(obs_list[1]["action_mask"])[0]
        neighbor_target_idx = None
        for idx in neighbor_indices:
            if idx > 1:  # 2+ 为邻居
                neighbor_target_idx = int(idx)
                break
        assert neighbor_target_idx is not None, "Vehicle1 无可用邻居目标"

        rem_data0_init = float(env.vehicles[0].task_dag.rem_data[sub0])
        rem_data1_init = float(env.vehicles[1].task_dag.rem_data[sub1])

        actions = [
            {"target": rsu_target_idx, "power": 1.0},
            {"target": neighbor_target_idx, "power": 1.0},
        ]
        env.step(actions)

        # Phase 1: 传输应推进（剩余数据下降）
        rem_data0_after = float(env.vehicles[0].task_dag.rem_data[sub0])
        rem_data1_after = float(env.vehicles[1].task_dag.rem_data[sub1])
        assert rem_data0_after < rem_data0_init or rem_data1_after < rem_data1_init

        rsu_task = env.rsus[0].active_task_manager.get_task(v0_id, sub0)
        veh0_task = env.vehicles[0].active_task_manager.get_task(v1_id, sub1)
        handover_seen = rsu_task is not None or veh0_task is not None

        # 后续仅推进物理过程
        prev_rsu_rem = rsu_task.rem_comp if rsu_task else None
        prev_veh0_rem = veh0_task.rem_comp if veh0_task else None
        completed = False

        for _ in range(1, 50):
            obs_list, rewards, terminated, truncated, info = env.step([{} for _ in env.vehicles])

            # Phase 2: Handover 检测
            rsu_task = env.rsus[0].active_task_manager.get_task(v0_id, sub0)
            veh0_task = env.vehicles[0].active_task_manager.get_task(v1_id, sub1)
            if rsu_task or veh0_task:
                handover_seen = True

            # Phase 3: 计算进度增加（剩余计算量减少）
            if rsu_task and prev_rsu_rem is not None:
                assert rsu_task.rem_comp <= prev_rsu_rem + 1e-6
                prev_rsu_rem = rsu_task.rem_comp
            if veh0_task and prev_veh0_rem is not None:
                assert veh0_task.rem_comp <= prev_veh0_rem + 1e-6
                prev_veh0_rem = veh0_task.rem_comp

            # Phase 4: 完成检测
            if env.vehicles[0].task_dag.status[sub0] == 3 and env.vehicles[1].task_dag.status[sub1] == 3:
                completed = True
                break

        assert handover_seen, "未检测到任务切换至执行节点"
        assert completed, "任务未在生命周期内完成"

        # 运行时长输出（基于环境时间与步数）
        print(f"Episode runtime: {env.time:.3f}s, steps={env.steps}")
        print("SYSTEM INTEGRATION PASSED")
    finally:
        # 恢复配置
        Cfg.NUM_VEHICLES = orig["NUM_VEHICLES"]
        Cfg.NUM_RSU = orig["NUM_RSU"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
        Cfg.MAX_STEPS = orig["MAX_STEPS"]
