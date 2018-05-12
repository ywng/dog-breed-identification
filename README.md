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
Connect to prince cluster:
```bash
$ssh gw.hpc.nyu.edu
$ssh prince
$cd /scratch/$User/

$sbatch run-transfer-densenet201.s 
$squeue -u ywn202
$cat slurm-6160773
```
