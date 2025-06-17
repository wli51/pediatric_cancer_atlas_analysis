import sys
import pathlib
import subprocess
import yaml

# Find repo root dynamically with
# subprocess run of git --show-toplevel command
def get_repo_root():
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        check=True,
        text=True
    ).stdout.strip()

ROOT_PATH = pathlib.Path(get_repo_root()).absolute()

CONFIG_FILE = ROOT_PATH / 'config.yml'
assert CONFIG_FILE.exists(), f"File not found: {CONFIG_FILE}"

with open(ROOT_PATH / "config.yml", "r") as file:
    config = yaml.safe_load(file)

SOFTWARE_PATH = pathlib.Path(config['paths']['software_path'])
assert SOFTWARE_PATH.exists(), f"Directory not found: {SOFTWARE_PATH}"
sys.path.append(str(SOFTWARE_PATH))
try:
    import virtual_stain_flow
except ImportError:
    raise ImportError(f"Module 'virtual_stain_flow' not found. Ensure it is present in {SOFTWARE_PATH}")

LOADDATA_FILE_PATH = ROOT_PATH \
    / '0.data_preprocessing' / 'data_split_loaddata' / 'loaddata_train.csv'
assert LOADDATA_FILE_PATH.exists(), f"File not found: {LOADDATA_FILE_PATH}"

PROFILING_DIR = pathlib.Path(config['paths']['pediatric_cancer_atlas_profiling_path'])
assert PROFILING_DIR.exists(), f"Directory not found: {PROFILING_DIR}"

SC_FEATURES_DIR = pathlib.Path(config['paths']['sc_features_path'])
assert SC_FEATURES_DIR.exists(), f"Directory not found: {SC_FEATURES_DIR}"

INPUT_CHANNEL_NAMES = config['data']['input_channel_keys']
TARGET_CHANNEL_NAMES = config['data']['target_channel_keys']