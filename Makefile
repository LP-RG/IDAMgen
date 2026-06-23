
PY := python3.8
ENV_NAME := .venv
DELETION_DELAY ?= 10

# computed

ACTIV_ENV := . $(ENV_NAME)/bin/activate
IN_ENV := $(ACTIV_ENV) &&

py_init:
	@echo "\n[[ creating python environment if absent ]]"
	$(PY) -m venv $(ENV_NAME)

py_dep: py_init
	@echo "\n[[ installing/upgrading python dependencies ]]"
	$(IN_ENV) python3 -m pip install --upgrade pip setuptools wheel
	$(IN_ENV) pip install --upgrade -r requirements.txt
	$(IN_ENV) pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
	$(IN_ENV) pip install --no-build-isolation ./src/extension_cpp/.

setup: py_init py_dep 

rm_cache:
	@echo "\n[[ removing all pycache folders ]]"
	find . -name __pycache__ -prune -print -exec rm -rf {} \;

rm_temp:
	@echo "\n[[ removing all temporary files ]]"
	find . -name "*.tmp" -print -exec rm -rf {} \;

rm_pyenv:
	@echo "\n[[ removing the virtual python environment ]]"
	rm -rf $(ENV_NAME)

clean: rm_cache rm_temp rm_pyenv
	@echo "\n[[ all clean! ]]"