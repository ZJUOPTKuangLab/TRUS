# TRUS
TRUS(Transformer-based Reconstruction for Ultr-fast SOFI imaging) is a deep learning method for ultra-fast SOFI reconstruction.
# Environment
- Windows 10
- CUDA 11.2
- Python 3.8.10
- numpy 1.24.1
- opencv 4.10.0
- scikit-image 0.21.0
- pytorch-gpu 1.9.1
- GPU GeForce RTX 2080 Ti
# File Description
`./data` is the default path for training data
`./data/wf` The augmented training widefield image will be saved here
`./data/20f` The augmented training sofi image (reconstructed from 20 frames) will be saved here
`./data/gt` The augmented training sofi image (reconstrcuted from 3000frames) will be saved here
`./weights` place pre-trained TRUS models here for testing
`./models` place network architecture and dataloader
# Usage
## Test 
The file test.py is designed for performing inference on few-shot SOFI and widefield image pairs. The variable wf_uafpath specifies the path to the widefield image, while uaf_path specifies the path to the SOFI image.

## Training
The TRUS can be trained by the file <train.py> with datasets containing widefield and SOFI image pairs. The path of dataset should be changed according to the actual configuration.
