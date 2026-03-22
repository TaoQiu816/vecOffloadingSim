# LR Critic Sweep Report

## Config Summary
- lr_critic=2e-4: lr_actor=0.0002, lr_critic=0.0002, episodes=1000, best50_task_sr=0.9580
- lr_critic=3e-4: lr_actor=0.0002, lr_critic=0.0003, episodes=1000, best50_task_sr=0.9580
- lr_critic=5e-4: lr_actor=0.0002, lr_critic=0.0005, episodes=1000, best50_task_sr=0.9630

## Tail-100 Highlights
- Average Reward: winner=lr_critic=2e-4
  - lr_critic=2e-4: 0.010839
  - lr_critic=3e-4: 0.010625
  - lr_critic=5e-4: 0.009098
- Task Success Rate: winner=lr_critic=2e-4
  - lr_critic=2e-4: 0.928500
  - lr_critic=3e-4: 0.896000
  - lr_critic=5e-4: 0.913000
- Subtask Success Rate: winner=lr_critic=2e-4
  - lr_critic=2e-4: 0.990413
  - lr_critic=3e-4: 0.983141
  - lr_critic=5e-4: 0.957497
- Deadline Miss Rate: winner=lr_critic=2e-4
  - lr_critic=2e-4: 0.071500
  - lr_critic=3e-4: 0.104000
  - lr_critic=5e-4: 0.087000
- Average RSU Queue: winner=lr_critic=2e-4
  - lr_critic=2e-4: 1.260782
  - lr_critic=3e-4: 1.514313
  - lr_critic=5e-4: 7.130853
- Average Power: winner=lr_critic=5e-4
  - lr_critic=2e-4: 0.451783
  - lr_critic=3e-4: 0.453116
  - lr_critic=5e-4: 0.435519
- Approx KL: winner=lr_critic=3e-4
  - lr_critic=2e-4: 0.019096
  - lr_critic=3e-4: 0.017056
  - lr_critic=5e-4: 0.021099
- Entropy: winner=n/a
  - lr_critic=2e-4: 1.426623
  - lr_critic=3e-4: 1.111896
  - lr_critic=5e-4: 1.430001

## Baseline Task-SR
- lr_critic=2e-4: Local-Only=0.8530, Greedy=0.8610, EFT=0.5740, CP-EFT=0.6020
- lr_critic=3e-4: Local-Only=0.8530, Greedy=0.8610, EFT=0.5740, CP-EFT=0.6020
- lr_critic=5e-4: Local-Only=0.8530, Greedy=0.8610, EFT=0.5740, CP-EFT=0.6020

## Key Findings
- Highest tail-100 task success: lr_critic=2e-4; lowest deadline miss: lr_critic=2e-4.
- Lowest tail-100 RSU queue: lr_critic=2e-4; lowest power: lr_critic=5e-4.
- Best tail-100 reward: lr_critic=2e-4.
