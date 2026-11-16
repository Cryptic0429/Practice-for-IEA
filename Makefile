PYTHON ?= python3

.PHONY: install verify verify-strict test test-runtime studio report clean-report

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

verify:
	$(PYTHON) scripts/project/assets.py verify

verify-strict:
	$(PYTHON) scripts/project/assets.py verify --require-weights

test:
	$(PYTHON) -m unittest discover -s scripts/project -p 'test_*.py' -v
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts $(PYTHON) -m unittest discover -s scripts/iea_agent -p 'test_*.py' -v

test-runtime:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts $(PYTHON) -m unittest tests.test_runtime iea_studio.test_app -v

studio:
	bash scripts/iea_studio/launch.sh

report:
	bash reports/mainline-technical-20261113/build.sh

clean-report:
	$(PYTHON) -c "from pathlib import Path; import shutil; p=Path('tmp/pdfs/mainline-technical-20261113'); shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else None"
