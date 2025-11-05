# topology-and-robustness-in-DNNs
We quantify how the topology of neural activations evolves across layers and test whether these topological trajectories predict model robustness under structured input perturbations.

## Setup and dependencies

This project manages its Python environment with a simple helper script and `requirements.txt`.

### Prerequisites
- Python 3.x installed and available as `python3` (macOS/Linux) or `python` (Windows Git Bash)
- `bash` shell

### First-time installation
```bash
chmod +x manage_dependensies.sh
./manage_dependensies.sh install
```
This will:
- Create a fresh virtual environment in `venv/` (removes an existing one if present)
- Upgrade `pip`
- Install packages from `requirements.txt`

Activate the virtual environment after installation:
- macOS/Linux:
  ```bash
  source venv/bin/activate
  ```
- Windows (Git Bash):
  ```bash
  source venv/Scripts/activate
  ```

### Update dependencies later
```bash
./manage_dependensies.sh update
```
This upgrades `pip` and updates packages to the versions specified in `requirements.txt`.

### Help
```bash
./manage_dependensies.sh --help
```
Also available as `-h` or `help`.

**If you still run into issues contact me (Sven)**

### Notes
- Dependencies are listed in `requirements.txt`.
- The `install` command recreates the `venv/` from scratch; use `update` to keep your existing environment and just refresh packages.
