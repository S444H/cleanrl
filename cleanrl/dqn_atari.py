# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import random
import time

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 日志写入器，用于训练指标可视化

import tyro  # 现代命令行解析库

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,  # 将奖励裁剪到[-1, 1] (论文)
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,  # 跳帧并取最大帧 (论文)
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer

from dataclasses import dataclass, fields
import os


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]  # 实验名称
    seed: int = 1  # 实验随机种子
    torch_deterministic: bool = True  # 是否开启 PyTorch 确定性模式
    cuda: bool = True  # 是否使用 CUDA（GPU）

    track: bool = False  # 是否使用 Weights & Biases 追踪实验
    wandb_project_name: str = "cleanRL"  # W&B 项目名称
    wandb_entity: str = None  # W&B 团队或用户名

    capture_video: bool = False  # 是否录制智能体运行视频（保存到 videos 文件夹）
    save_model: bool = False  # 是否保存训练好的模型
    upload_model: bool = False  # 是否上传模型到 Hugging Face Hub
    hf_entity: str = ""  # Hugging Face 用户名或组织名

    # 算法相关参数
    env_id: str = "BreakoutNoFrameskip-v4"  # 环境 ID,NoFrame方便自己控制帧跳过 (论文)
    total_timesteps: int = 10000000  # 总训练步数
    learning_rate: float = 1e-4  # 优化器学习率
    num_envs: int = 1  # 并行环境数量
    buffer_size: int = 100000  # 经验回放缓冲区大小:1000000
    gamma: float = 0.99  # 折扣因子
    tau: float = 1.0  # 目标网络更新系数（1.0 表示硬更新）
    target_network_frequency: int = 1000  # 目标网络更新频率（步数）
    batch_size: int = 32  # 训练批量大小
    start_e: float = 1  # epsilon-greedy 初始探索率
    end_e: float = 0.01  # epsilon-greedy 最终探索率
    exploration_fraction: float = 0.10  # 从 start_e 到 end_e （探索阶段）所占总训练步数比例
    learning_starts: int = 80000  # 训练开始步数（经验池积累后开始训练）
    train_frequency: int = 4  # 训练频率（每隔多少步训练一次）

    def print_args(self):
        """打印当前配置参数"""
        print("=" * 40)
        print(f"{'参数名称':<30} | {'值'}")
        print("-" * 40)
        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            print(f"{name:<30} | {value}")
        print("=" * 40)


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # 保持论文网络结构：
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


# 线性衰减 epsilon
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    # 将所有超参数以 Markdown 表格的形式写入 TensorBoard 文本面板
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding效果一致性设定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # network setup
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    # replaybuffer setup
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    # ALGO LOGIC: start game
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: action logic
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,  # 探索迭代次数
                                  global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # 每轮游戏结束后，把训练数据写入 TensorBoard，以便在训练过程中实时绘制曲线
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # save data to reply buffer
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # ALGO LOGIC: training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:  # 训练频率
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # 每 100轮 把相关数据写入 TensorBoard
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network/1000，软更新
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:  # 默认不保存
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
