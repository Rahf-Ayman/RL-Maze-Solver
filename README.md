## Installation:
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify Gymnasium is working
python -c "import gymnasium; print(gymnasium.__version__)"

# Verify PyTorch
python -c "import torch; print(torch.__version__)"
```