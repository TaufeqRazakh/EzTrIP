import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / 'test_inputs/config-train.yaml'

def test_config_file_exists():
	print(CONFIG_PATH)
	assert CONFIG_PATH.exists(), "Expected config file, found none"

def test_config_file_readable():
	with CONFIG_PATH.open(mode='r') as f:
		config = yaml.safe_load(f)
		assert "epochs" in config, "Expected epochs in config file, missing values"
		print(config['epochs'])
		architecture = config['architecture']
		assert "embedding_dim" in architecture, "Expected velues for architecture, found none"
		print(architecture)
