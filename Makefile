PYTHON ?= python3

.PHONY: verify test test-runtime report clean-report
verify:
	$(PYTHON) scripts/project/assets.py verify

test:
	$(PYTHON) -m unittest discover -s scripts/project -p 'test_*.py' -v
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts $(PYTHON) -m unittest discover -s scripts/iea_agent -p 'test_*.py' -v

test-runtime:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts $(PYTHON) -m unittest tests.test_runtime iea_studio.test_app -v

report:
	bash reports/mainline-technical-20261113/build.sh

clean-report:
	$(PYTHON) -c "from pathlib import Path; import shutil; p=Path('tmp/pdfs/mainline-technical-20261113'); shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else None"
