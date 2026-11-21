# Run Aloha Sim

## Interactive 3D inspection

From the project root, run `env -u PYTHONPATH .venv/bin/python scripts/view_simulation.py`.
Select **ALOHA Sim** and choose cube transfer or bimanual insertion. The scene opens in the full MuJoCo interface and starts paused.
The default `pi0_aloha_sim` checkpoint is for cube transfer; insertion inference requires a compatible task checkpoint.
See [the inspector guide](../../docs/simulation_inspector.md) for scene selection, reset and model controls.

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e packages/openpi-client

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

Note: If you are seeing EGL errors, you may need to install the following dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env ALOHA_SIM
```
