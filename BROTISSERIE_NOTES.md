# Brotisserie PPO Investigation Notes

## Operating assumption

This fork is an agentic coding workspace. Hermes is expected to do the implementation work directly: create branches, edit code, run tests/benchmarks, inspect failures, commit changes when appropriate, and report verified outcomes rather than only giving instructions.

## Project goal

Build and benchmark a better PPO baseline/default recipe that requires less domain-specific hyperparameter expertise while remaining credible against existing Stable-Baselines3 baselines.

The core research question:

> Can one robust PPO default configuration or small set of automatic heuristics perform reasonably across diverse environment families compared with SB3 defaults and per-environment tuned RL Zoo configs?

## Hardware target

Primary hardware target is an AMD Radeon RX 7900 GRE on ROCm. Prefer PyTorch-compatible approaches and avoid making CUDA-only assumptions.

## Baseline stack

- Stable-Baselines3 fork as the benchmark-compatible implementation anchor
- RL Baselines3 Zoo as the tuned per-environment best-case comparison
- Gymnasium/Farama environments as the common API
- CleanRL as a readable reference for algorithm surgery when needed
- TensorBoard or Weights & Biases for logging

## Comparison tiers

1. **SB3 default PPO**
   - Current default PPO behavior/configs from upstream SB3.

2. **RL Zoo tuned PPO**
   - Per-environment tuned hyperparameters from RL Baselines3 Zoo.
   - Represents expert-tuned best-case or near-best-case baseline.

3. **Brotisserie PPO**
   - A proposed improved default PPO recipe.
   - Should avoid env-specific hand tuning where possible.
   - May use simple automatic rules based on action space, observation modality, rollout size, or training health metrics.

## Initial hackable areas in SB3 PPO

Likely first places to modify:

- `stable_baselines3/ppo/ppo.py`
  - `PPO.train()` for loss, clipping, target KL, advantage normalization, entropy/value coefficients, optimizer behavior.

- `stable_baselines3/common/on_policy_algorithm.py`
  - rollout collection only if needed.

- `stable_baselines3/common/buffers.py`
  - advantage/return behavior or minibatch semantics only if needed.

- `stable_baselines3/common/policies.py`
  - architecture/init/default policy behavior.

Prefer starting with minimal changes inside `PPO.train()` and constructor defaults before touching rollout collection or buffers.

## Candidate Brotisserie PPO ideas

Early candidates to test:

- Adaptive learning-rate schedule based on approximate KL or clip fraction
- Adaptive clip range based on policy update health
- Better target KL defaults
- Safer entropy coefficient defaults for discrete vs continuous action spaces
- Automatic batch/minibatch heuristics from `n_envs`, `n_steps`, and env family
- More robust advantage normalization behavior
- Value function clipping default review
- Better default network architecture/init for continuous control vs image/discrete domains
- Standardized health logging: KL, clip fraction, explained variance, entropy, value loss, policy loss, gradient norm

## Candidate environment suite

Start small, then expand.

### Smoke/debug

- `CartPole-v1`
- `Pendulum-v1`
- `MountainCarContinuous-v0`

### Box2D

- `LunarLander-v3`
- `BipedalWalker-v3`

### MuJoCo

- `HalfCheetah-v5`
- `Hopper-v5`
- `Walker2d-v5`
- `Ant-v5`

### Atari / visual discrete control

- `PongNoFrameskip-v4`
- `BreakoutNoFrameskip-v4`

## Measurement priorities

Track more than final return:

- Mean return and std across seeds
- Sample efficiency at fixed environment steps
- Wall-clock time
- Environment steps/sec
- GPU utilization where available
- Failure/instability rate across seeds
- PPO health metrics: approximate KL, clip fraction, entropy, explained variance, gradient norm if added

## Immediate next steps

1. Create a working branch such as `brotisserie-ppo`.
2. Set up a local dev environment suitable for ROCm PyTorch.
3. Run minimal SB3 tests/import checks.
4. Add a benchmark harness or scripts for default PPO vs Brotisserie PPO.
5. Establish first small benchmark matrix with 3 seeds on smoke/debug envs.
6. Only then start modifying PPO defaults/logic.

## Repo state at notes creation

- Fork: `https://github.com/markschatza/stable-baselines3`
- Local path: `/home/marks/c/git/stable-baselines3`
- Upstream: `https://github.com/DLR-RM/stable-baselines3`
- Base branch: `master`
- Starting HEAD observed: `8908708 Release v2.9.0 (#2262)`
