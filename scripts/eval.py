import argparse, json
from dataclasses import asdict
import torch
from rsl_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
import mjlab.tasks  # noqa: F401
import in_hand_rotation_mjlab.tasks  # noqa: F401

HELD_OUT = {"friction_range": (0.3, 2.2), "mass_range": (0.3, 2.5)}
HELD_OUT_EVENTS = ["dr_shared_contact_friction", "dr_cube_mass"]


def prep(agent_cfg):
  d = asdict(agent_cfg)
  for k in ("actor", "critic"):
    m = d.get(k)
    if isinstance(m, dict) and m.get("class_name", "MLPModel") != "CNNModel":
      m.pop("cnn_cfg", None)
  return d


def apply_held_out(env_cfg):
  for name in HELD_OUT_EVENTS:
    ev = env_cfg.events[name]
    hit = False
    for k, v in HELD_OUT.items():
      if k in ev.params:
        ev.params[k] = v
        hit = True
    assert hit, f"{name}: {list(ev.params)}"


def main():
  p = argparse.ArgumentParser()
  p.add_argument("task")
  p.add_argument("ckpt")
  p.add_argument("mode", choices=["nominal", "held_out"])
  p.add_argument("seed", type=int)
  p.add_argument("out")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--num-steps", type=int, default=1200)
  a = p.parse_args()

  configure_torch_backends()
  device = "cuda:0"
  torch.manual_seed(a.seed)
  env_cfg = load_env_cfg(a.task)
  agent_cfg = load_rl_cfg(a.task)
  env_cfg.scene.num_envs = a.num_envs
  if hasattr(env_cfg, "seed"):
    env_cfg.seed = a.seed
  if a.mode == "held_out":
    apply_held_out(env_cfg)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  max_len = getattr(env, "max_episode_length", 400)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(a.task) or OnPolicyRunner
  runner = runner_cls(env, prep(agent_cfg), device=device)
  runner.load(a.ckpt, map_location=device)
  policy = runner.get_inference_policy(device=device)

  env.reset()  # start every env from a clean reset
  obs = env.get_observations()
  ep_len = torch.zeros(a.num_envs, device=device)
  ep_idx = torch.zeros(a.num_envs, dtype=torch.long, device=device)
  st = {"first_done": 0, "first_early": 0, "later_done": 0, "later_early": 0}
  lengths, n_done, n_early = [], 0, 0
  logs, wsum = {}, 0

  for _ in range(a.num_steps):
    with torch.inference_mode():
      obs, rew, dones, extras = env.step(policy(obs))
    ep_len += 1
    d = dones.bool()
    if d.any():
      to = extras.get("time_outs")
      early = d & ~to.bool() if to is not None else d & (ep_len < max_len)
      first = d & (ep_idx == 0)
      later = d & (ep_idx > 0)
      st["first_done"] += int(first.sum())
      st["first_early"] += int((early & first).sum())
      st["later_done"] += int(later.sum())
      st["later_early"] += int((early & later).sum())
      n_done += int(d.sum())
      n_early += int(early.sum())
      lengths.append(ep_len[d].clone())
      ep_len[d] = 0
      ep_idx[d] += 1
      # weight logged episode metrics by the number of envs that reset this step
      w = int(d.sum())
      for k, v in extras.get("log", {}).items():
        if k.startswith("Episode_Metrics/"):
          logs[k] = logs.get(k, 0.0) + float(v) * w
      wsum += w

  L = torch.cat(lengths)
  res = {
    "mode": a.mode,
    "ckpt": a.ckpt,
    "num_envs": a.num_envs,
    "episodes": n_done,
    "mean_episode_length": L.mean().item(),
    "early_rate": n_early / max(n_done, 1),
    "first_episode_fail": st["first_early"] / max(st["first_done"], 1),
    "later_episode_fail": st["later_early"] / max(st["later_done"], 1),
    **st,
    **{k: v / max(wsum, 1) for k, v in logs.items()},
  }
  json.dump(res, open(a.out, "w"), indent=2)
  print(json.dumps(res, indent=2))
  env.close()


if __name__ == "__main__":
  main()