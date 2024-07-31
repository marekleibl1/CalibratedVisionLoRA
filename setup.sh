
# Create and activate a virtual env 
python3 -m venv venv && source venv/bin/activate

# Install all Python dependencies 
pip install -U pip && pip install git+https://github.com/aleximmer/Laplace.git@508843d torch==2.2 numpy==1.26.3 matplotlib netcal ipykernel  safetensors

# Make the created venv available in jupyter notebooks
python -m ipykernel install --user --name=myenv --display-name "venv"
