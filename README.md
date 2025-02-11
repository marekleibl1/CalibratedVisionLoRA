

# Calibration of LoRA Vision Models

This project applies the Laplace approximation to calibrate a Vision Transformer (ViT) model fine-tuned using Low-Rank Adaptation (LoRA). 

## Key Components
- **LoRA Fine-Tuning:** Introduces a small set of trainable parameters to efficiently adapt models.
- **Laplace Method:** Approximates the posterior distribution of model parameters with a Gaussian distribution.
- **Random Projections:** Further reduce parameters for using Full Laplace.

## Results
- **Calibration Improvements:** Significant reduction in expected calibration error (ECE).
- **Model Accuracy:** Maintained with calibrated models.

## Requirements
The code was tested with the following setup: Ubuntu 22.04, A40 GPU with 46GB VRAM, cuda 12.1, Python 3.10. 

The code will likely run on 12GB VRAM GPU if batch size is reduced. 

## Setup Instructions
- Clone the repository
- Run or manually execute commands from the setup script: **setup.sh**
- This will create a virual environment and install all required Python packages. 






