"""Patch installed Brax PPO to avoid jax.pmap for single-device ROCm smoke tests.

This is a local Brotisserie debugging patch for MuJoCo Playground on AMD ROCm.
It modifies the active virtualenv's brax.training.agents.ppo.train module in
place. Re-run after reinstalling brax.
"""
from __future__ import annotations

from pathlib import Path

import brax.training.agents.ppo.train as ppo_train

path = Path(ppo_train.__file__)
text = path.read_text()

repls = [
    (
        "def _unpmap(v):\n"
        "  # Avoid degraded performance under the new jax.pmap.\n"
        "  return jax.tree_util.tree_map(\n"
        "      lambda x: x.addressable_shards[0].data.squeeze(0), v\n"
        "  )\n",
        "def _unpmap(v):\n"
        "  # Brotisserie single-device ROCm patch: values are no longer pmap-\n"
        "  # replicated, so return them unchanged.\n"
        "  return v\n",
    ),
    (
        "  loss_and_pgrad_fn = gradients.loss_and_pgrad(\n"
        "      loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True\n"
        "  )\n",
        "  loss_and_pgrad_fn = gradients.loss_and_pgrad(\n"
        "      loss_fn, pmap_axis_name=None, has_aux=True\n"
        "  )\n",
    ),
    (
        "          pmap_axis_name=_PMAP_AXIS_NAME,\n",
        "          pmap_axis_name=None,\n",
    ),
    (
        "  key_envs = jax.random.split(key_env, num_envs // process_count)\n"
        "  key_envs = jnp.reshape(\n"
        "      key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]\n"
        "  )\n"
        "  if local_devices_to_use > 1 or use_pmap_on_reset:\n"
        "    reset_fn_ = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)\n"
        "    env_state = reset_fn_(key_envs)\n"
        "    reset_fn = jax.pmap(\n"
        "        reset_fn_donated_env_state,\n"
        "        axis_name=_PMAP_AXIS_NAME,\n"
        "        donate_argnums=(0,),\n"
        "    )\n"
        "  else:\n"
        "    reset_fn_ = jax.jit(jax.vmap(env.reset))\n"
        "    env_state = reset_fn_(key_envs)\n"
        "    reset_fn = jax.jit(\n"
        "        reset_fn_donated_env_state, donate_argnums=(0,), keep_unused=True\n"
        "    )\n"
        "\n"
        "  # Discard the batch axes over devices and envs.\n"
        "  obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)\n",
        "  key_envs = jax.random.split(key_env, num_envs // process_count)\n"
        "  reset_fn_ = jax.jit(env.reset)\n"
        "  env_state = reset_fn_(key_envs)\n"
        "  reset_fn = jax.jit(\n"
        "      reset_fn_donated_env_state, donate_argnums=(0,), keep_unused=True\n"
        "  )\n"
        "\n"
        "  # Discard the batch axis over envs.\n"
        "  obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], env_state.obs)\n",
    ),
    (
        "  training_epoch = jax.pmap(\n"
        "      training_epoch,\n"
        "      axis_name=_PMAP_AXIS_NAME,\n"
        "      donate_argnums=(\n"
        "          0,\n"
        "          1,\n"
        "      ),\n"
        "  )\n",
        "  training_epoch = jax.jit(\n"
        "      training_epoch,\n"
        "      donate_argnums=(\n"
        "          0,\n"
        "          1,\n"
        "      ),\n"
        "  )\n",
    ),
    (
        "  # Compatibility shim for JAX >= 0.10, where jax.device_put_replicated\n"
        "  # was removed. This local benchmark uses one ROCm device, so adding the\n"
        "  # leading pmap replica axis and device_put() is equivalent for our case.\n"
        "  if hasattr(jax, 'device_put_replicated'):\n"
        "    training_state = jax.device_put_replicated(\n"
        "        training_state, jax.local_devices()[:local_devices_to_use]\n"
        "    )\n"
        "  else:\n"
        "    if local_devices_to_use != 1:\n"
        "      raise NotImplementedError(\n"
        "          'Local JAX>=0.10 compatibility shim only handles one device.'\n"
        "      )\n"
        "    training_state = jax.device_put(\n"
        "        jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), training_state),\n"
        "        jax.local_devices()[0],\n"
        "    )\n",
        "  # Brotisserie single-device ROCm patch: keep TrainingState unreplicated.\n"
        "  training_state = jax.device_put(training_state, jax.local_devices()[0])\n",
    ),
    (
        "      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)\n"
        "      (training_state, env_state, training_metrics) = (\n"
        "          training_epoch_with_timing(training_state, env_state, epoch_keys)\n"
        "      )\n",
        "      (training_state, env_state, training_metrics) = (\n"
        "          training_epoch_with_timing(training_state, env_state, epoch_key)\n"
        "      )\n",
    ),
    (
        "      key_envs = jax.vmap(\n"
        "          lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)\n"
        "      )(key_envs, key_envs.shape[1])\n",
        "      key_envs = jax.random.split(key_envs[0], key_envs.shape[0])\n",
    ),
    (
        "  # If there was no mistakes the training_state should still be identical on all\n"
        "  # devices.\n"
        "  pmap.assert_is_replicated(training_state)\n",
        "  # Brotisserie single-device ROCm patch: no replicated pmap state.\n",
    ),
    (
        "  pmap.synchronize_hosts()\n",
        "  # Brotisserie single-device ROCm patch: no host synchronization needed.\n",
    ),
]

for old, new in repls:
    if old not in text:
        print(f"SKIP missing block starting: {old.splitlines()[0]!r}")
    else:
        text = text.replace(old, new)

path.write_text(text)
print(path)
