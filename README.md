# Dog Breed Identification

### Kaggle Competition & Data
Download the data and prepare the data in below directories using data/data_prep.py

https://www.kaggle.com/c/dog-breed-identification

    ├── data
    │   ├── labels.csv
    │   ├── sample_submission.csv
    │   ├── test
    |   |     | x
    │   ├── train
    |   |     | .... classes folders
    │   └── valid
    |   |     | .... classes folders
    │   ├── training_images

### How to Run
#### Locally
After finishing the data setup preparation, you can just simple run your jupyter notebook or 
```bash
$python3 transfer-xyz.py
```
#### HPC
Connect to prince cluster and schedule your job to run by slurm.
```bash
$ssh gw.hpc.nyu.edu
$ssh prince
$cd /scratch/$User/

$sbatch run-transfer-densenet201.s 
$squeue -u ywn202
$cat slurm-6160773
```
Here are some instructions from HPC for using PyTorch with a virtual environment:

To use Python 3.6.3 with PyTorch 0.2.0_3, I’m working in the folder /home/wang/pyenv
 
(1) To create a new folder
 
 mkdir py3.6.3
 
(2) module load pytorch/python3.6/0.2.0_3
 
(3) virtualenv --system-site-packages py3.6.3
 
(4) source py3.6.3/bin/activate
 
(5) pip install opencv-python Pillow pywavelets scikit-learn
 
Now every time after run
 
module load pytorch/python3.6/0.2.0_3
source /home/wang/pyenv/py3.6.3/bin/activate
 
you’ll use this environment. 
