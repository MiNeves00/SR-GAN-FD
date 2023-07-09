# SR-GAN-FD
Super-Resolution (SR) Generative Adversarial Networks (GAN) methods for Fluid Dynamics (FD) images.

FD-SR is a project part of a Master's Thesis at FEUP partnered with LIACC (Artificial Intelligence and Computer Science Lab).

The Thesis is named **Application of Novel Techniques in
Super Resolution GANs for Fluid
Dynamics**.


The goal is to use super-resolution GAN techniques to enhance the quality of images obtained from Computational Fluid Dynamics (CFD) simulations and Experimental Fluid Dynamics.

Models explored include the ESRGAN, BSRGAN, Real-ESRGAN (Discriminator only), and the A-ESRGAN.

## Configuration
The hyperparameters and training configurations can be adjusted in the respective file. For example for the BSRGAN it is the `bsrgan_config.py` file. 

This configuration file sets various parameters, including:
- Train, Development, Test datasets
- Degradation process parameters
- Random seed for maintaining reproducible results
- The device for training (default: GPU)
- Whether only the Y channel image data should be verified when evaluating the SR model
- Model architecture names
- Discriminator and generator configurations
- MLflow experiment details
- Training parameters like dataset address, batch size, number of epochs, learning rate, etc
- Testing parameters like directory for ground truth images, whether to save images and metrics, etc.

You can find the BSRGAN configuration file [here](BSRGAN/bsrgan_config.py).

## Training
To train the models, you need to navigate into the respective model's folder. For example, to train the BSRGAN model, follow these steps:

1. Change your current directory to the BSRGAN folder: `cd BSRGAN`
2. Modify the mode to `train` in the configuration script and choose if a pretrained model should be loaded (from a MLFlow run or from a file): `python bsrgan_config.py`
3. Run the training script: `python train_bsrgan.py`

## Testing
To test the models, you need to run the test script inside the respective model's folder. For example, to test the BSRGAN model, follow these steps:

1. Change your current directory to the BSRGAN folder: `cd BSRGAN`
2. Modify the mode to `test` in the configuration script and provide the path to the model: `python bsrgan_config.py`
3. Run the testing script: `python test_bsrgan.py`
4. Resulting metrics and images are presented in the respective MLFlow folder: `cd mlruns/{experiment_id}/{run_id}`

For detailed instructions and more information about the project, see the comments in the provided scripts.
