# kaggle-lung-cancer

Kaggle Data Science Bowl 2017

Team: MDai (6th out of 1972)

## Requirements

- CUDA 8.0, cuDNN 5.1
- python 3.5+ (Anaconda)
- numpy 1.12.1
- scipy 0.19.0
- pandas 0.19.2
- scikit-image 0.13.0
- scikit-learn 0.18.1
- joblib 0.9.4
- pillow 4.0.0
- xgboost 0.6a2
- keras 1.2.2 (note: latest keras is 2)
- tensorflow (GPU) 1.0.0
- hyperopt 0.1
- pydicom built from master branch of git repo - bbaa74e9d02596afc03b924fe8ffbe7b95b6ff55 - 1.0.0a1 (not 0.9.9 from PyPI)
- h5py 2.7.0
- redis-py 2.10.5

See below to get an idea of how our environment was set up on an AWS p2.16xlarge instance:

```sh
# setup cuda
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
nvidia-smi
sudo nvidia-smi -pm 1
sudo nvidia-smi --auto-boost-default=0
sudo nvidia-smi -ac 2505,875

# setup cudnn
tar xzf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/include/* /usr/local/cuda/include
sudo cp cuda/lib64/* /usr/local/cuda/lib64

# setup anaconda and python libs
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/install/miniconda3
conda upgrade --all -y -q
conda install -y -q ipython joblib jupyter notebook pandas matplotlib numpy scipy requests scikit-image scikit-learn seaborn redis-py pyyaml tqdm h5py pillow
pip install --upgrade -q setuptools
pip install --upgrade -q hyperopt==0.1 ntfy==2.4.2 tensorflow-gpu==1.0.0 xgboost==0.6a2 Keras==1.2.2
git clone https://github.com/darcymason/pydicom.git && cd pydicom && python setup.py install && cd .. && rm -rf pydicom
export PATH=$HOME/install/miniconda3/bin:/usr/local/cuda/bin:$HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export KERAS_BACKEND=tensorflow

sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
```

## Predict

The prediction pipeline can eventually be configured as a single automated process (i.e., engineered for production), but to run it on our given datasets in a reasonable time we will run it stepwise according to the following steps. We ran these steps on a AWS p2.16xlarge instance which has 16 GPUs. All models and scripts are coded to run on a single GPU, so for the steps which involve splitting the data and running the splits across GPUs, we use the CUDA_VISIBLE_DEVICES flag to restrict each process to a single specific GPU. Otherwise, without this environment variable flag errors will result, due to tensorflow and the way the code is currently written.

**1. Preprocess DICOM studies**

Make sure data is available in the directories specified in `SETTINGS.json`.

```sh
# cwd: predict/
# --dataset may be {sample, stage1, stage2}
nohup python -u 01_preprocess.py --dataset stage2 > 01_preprocess.out.log 2>01_preprocess.err.log &
```

Processed data will be output to `data_proc/{dataset}/isotropic_volumes_1mm/`. Metadata containing original and new shapes and DICOM spacings will be output to `data_proc/{dataset}/isotropic_volumes_1mm.pkl`.

**2. Determine patient sex**

Ensure `sd01a.hdf5` weights file is available in the `weights/` directory. On multi-gpu machines, run script with single GPU specified with the CUDA_VISIBLE_DEVICES environment variable.

```sh
# cwd: predict/
# --dataset may be {sample, stage1, stage2}
CUDA_VISIBLE_DEVICES=0 nohup python -u 02_determine_sex.py --dataset stage2 > 02_determine_sex.out.log 2>02_determine_sex.err.log &
```

**3. Create ROI probability maps**

Ensure `m05a.hdf5` and `m09a.hdf5` weights files are available in the `weights/` directory. On multi-gpu machines, run script with single GPU per process, specified with the CUDA_VISIBLE_DEVICES environment variable. For example:

```sh
# cwd: predict/
# --dataset may be {sample, stage1, stage2}
# --config may be {50_16_2, 30_16_4}
# --split specifies # of processes to split studies into. This is based the # of GPUs available.
# --group specifies group # in split

CUDA_VISIBLE_DEVICES=0 nohup python -u 03_roi_probmaps.py --split 8 --group 1 --dataset stage2 --config 50_16_2 > 03_roi_probmaps.1a.out.log 2>03_roi_probmaps.1a.err.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u 03_roi_probmaps.py --split 8 --group 2 --dataset stage2 --config 50_16_2 > 03_roi_probmaps.2a.out.log 2>03_roi_probmaps.2a.err.log &
CUDA_VISIBLE_DEVICES=2 nohup python -u 03_roi_probmaps.py --split 8 --group 3 --dataset stage2 --config 50_16_2 > 03_roi_probmaps.3a.out.log 2>03_roi_probmaps.3a.err.log &
CUDA_VISIBLE_DEVICES=3 nohup python -u 03_roi_probmaps.py --split 8 --group 4 --dataset stage2 --config 50_16_2 > 03_roi_probmaps.4a.out.log 2>03_roi_probmaps.4a.err.log &
CUDA_VISIBLE_DEVICES=4 nohup python -u 03_roi_probmaps.py --split 8 --group 5 --dataset stage2 --config 50_16_2 > 03_roi_probmaps.5a.out.log 2>03_roi_probmaps.5a.err.log &
CUDA_VISIBLE_DEVICES=5 nohup python -u 03_roi_probmaps.py --split 8 --group 6 --dataset stage2 --config 50_16_2 > 03_roi_probmaps.6a.out.log 2>03_roi_probmaps.6a.err.log &
CUDA_VISIBLE_DEVICES=6 nohup python -u 03_roi_probmaps.py --split 8 --group 7 --dataset stage2 --config 50_16_2 > 03_roi_probmaps.7a.out.log 2>03_roi_probmaps.7a.err.log &
CUDA_VISIBLE_DEVICES=7 nohup python -u 03_roi_probmaps.py --split 8 --group 8 --dataset stage2 --config 50_16_2 > 03_roi_probmaps.8a.out.log 2>03_roi_probmaps.8a.err.log &

CUDA_VISIBLE_DEVICES=8 nohup python -u 03_roi_probmaps.py --split 8 --group 1 --dataset stage2 --config 30_16_4 > 03_roi_probmaps.1b.out.log 2>03_roi_probmaps.1b.err.log &
CUDA_VISIBLE_DEVICES=9 nohup python -u 03_roi_probmaps.py --split 8 --group 2 --dataset stage2 --config 30_16_4 > 03_roi_probmaps.2b.out.log 2>03_roi_probmaps.2b.err.log &
CUDA_VISIBLE_DEVICES=10 nohup python -u 03_roi_probmaps.py --split 8 --group 3 --dataset stage2 --config 30_16_4 > 03_roi_probmaps.3b.out.log 2>03_roi_probmaps.3b.err.log &
CUDA_VISIBLE_DEVICES=11 nohup python -u 03_roi_probmaps.py --split 8 --group 4 --dataset stage2 --config 30_16_4 > 03_roi_probmaps.4b.out.log 2>03_roi_probmaps.4b.err.log &
CUDA_VISIBLE_DEVICES=12 nohup python -u 03_roi_probmaps.py --split 8 --group 5 --dataset stage2 --config 30_16_4 > 03_roi_probmaps.5b.out.log 2>03_roi_probmaps.5b.err.log &
CUDA_VISIBLE_DEVICES=13 nohup python -u 03_roi_probmaps.py --split 8 --group 6 --dataset stage2 --config 30_16_4 > 03_roi_probmaps.6b.out.log 2>03_roi_probmaps.6b.err.log &
CUDA_VISIBLE_DEVICES=14 nohup python -u 03_roi_probmaps.py --split 8 --group 7 --dataset stage2 --config 30_16_4 > 03_roi_probmaps.7b.out.log 2>03_roi_probmaps.7b.err.log &
CUDA_VISIBLE_DEVICES=15 nohup python -u 03_roi_probmaps.py --split 8 --group 8 --dataset stage2 --config 30_16_4 > 03_roi_probmaps.8b.out.log 2>03_roi_probmaps.8b.err.log &
```

We must produce 2 sets of probability maps, with config settings `50_16_2` and `30_16_4`. Processed data will be output to `data_proc/{dataset}/nodule_detect_probmaps_50_16_2/` and `data_proc/{dataset}/nodule_detect_probmaps_30_16_4/`.

There are a lot of optimizations that can be done here, but this step currently takes quite a bit of time -- in total several days on a multi-GPU machine.

**4. Create cancer predictions and generate other features**

Ensure `m02a.hdf5`, `m04a.hdf5`, `m10a.hdf5`, `resnet2d09d.hdf5`, `resnet2d09e.hdf5`, `resnet2d09f.hdf5` weights files are available in the `weights/` directory. We must start redis in the background which will act as a simple job queue for the models server, which receives inference tasks to be run on the GPU for these neural networks.

Docker must be installed:

```sh
# install docker on ubuntu
sudo apt-get install -y wget
sudo wget -qO- https://get.docker.com/ | sh
sudo usermod -aG docker ubuntu
```

Start docker redis:

```sh
# cwd: predict/
./start-docker-redis.sh
```

Start models servers in the background, one process per GPU:

```sh
# cwd: predict/
CUDA_VISIBLE_DEVICES=0 nohup python -u models_server.py > models_server.1.out.log 2>models_server.1.err.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u models_server.py > models_server.2.out.log 2>models_server.2.err.log &
CUDA_VISIBLE_DEVICES=2 nohup python -u models_server.py > models_server.3.out.log 2>models_server.3.err.log &
CUDA_VISIBLE_DEVICES=3 nohup python -u models_server.py > models_server.4.out.log 2>models_server.4.err.log &
CUDA_VISIBLE_DEVICES=4 nohup python -u models_server.py > models_server.5.out.log 2>models_server.5.err.log &
CUDA_VISIBLE_DEVICES=5 nohup python -u models_server.py > models_server.6.out.log 2>models_server.6.err.log &
CUDA_VISIBLE_DEVICES=6 nohup python -u models_server.py > models_server.7.out.log 2>models_server.7.err.log &
CUDA_VISIBLE_DEVICES=7 nohup python -u models_server.py > models_server.8.out.log 2>models_server.8.err.log &
CUDA_VISIBLE_DEVICES=8 nohup python -u models_server.py > models_server.9.out.log 2>models_server.9.err.log &
CUDA_VISIBLE_DEVICES=9 nohup python -u models_server.py > models_server.10.out.log 2>models_server.10.err.log &
CUDA_VISIBLE_DEVICES=10 nohup python -u models_server.py > models_server.11.out.log 2>models_server.11.err.log &
CUDA_VISIBLE_DEVICES=11 nohup python -u models_server.py > models_server.12.out.log 2>models_server.12.err.log &
CUDA_VISIBLE_DEVICES=12 nohup python -u models_server.py > models_server.13.out.log 2>models_server.13.err.log &
CUDA_VISIBLE_DEVICES=13 nohup python -u models_server.py > models_server.14.out.log 2>models_server.14.err.log &
CUDA_VISIBLE_DEVICES=14 nohup python -u models_server.py > models_server.15.out.log 2>models_server.15.err.log &
CUDA_VISIBLE_DEVICES=15 nohup python -u models_server.py > models_server.16.out.log 2>models_server.16.err.log &
```

There are a number of hyperparameters to set for the cancer prediction and feature generation pipeline. The `configs.py` file contains 750 random such hyperparameter configurations. We will run this pipeline for all 750 specified configurations as follows:

```sh
# cwd: predict/
# --dataset may be {sample, stage1, stage2}
# --config will range from 1 to 750
nohup python -u 04_preds_and_features.py --dataset stage2 --config 1 > 04_preds_and_features.1.out.log 2>04_preds_and_features.1.err.log &
nohup python -u 04_preds_and_features.py --dataset stage2 --config 2 > 04_preds_and_features.2.out.log 2>04_preds_and_features.2.err.log &
nohup python -u 04_preds_and_features.py --dataset stage2 --config 3 > 04_preds_and_features.3.out.log 2>04_preds_and_features.3.err.log &
nohup python -u 04_preds_and_features.py --dataset stage2 --config 4 > 04_preds_and_features.4.out.log 2>04_preds_and_features.4.err.log &
# ...
nohup python -u 04_preds_and_features.py --dataset stage2 --config 749 > 04_preds_and_features.749.out.log 2>04_preds_and_features.749.err.log &
nohup python -u 04_preds_and_features.py --dataset stage2 --config 750 > 04_preds_and_features.750.out.log 2>04_preds_and_features.750.err.log &
```

An easy way to create these commands :

```py
# python
cmds = ''
for i in range(1, 751):
  cmds += 'nohup python -u 04_preds_and_features.py --dataset stage2 --config {} > 04_preds_and_features.{}.out.log 2>04_preds_and_features.{}.err.log &\n'.format(i,i,i)
print(cmds)
```

To check progress, one can inspect the number of items in the `processing` set in redis:

```py
# python
import redis
client = redis.StrictRedis(decode_responses=True)
print(client.scard('processing'))
```

There will be individual cancer predictions at `predictions/preds_{1-750}.csv`, and additional features at `data_proc/{dataset}/features/config_{1-750}_features.csv`. There must be 750 preds_$i.csv and 750 config_$i_features.csv files, each with number of lines equal to the number of samples in the dataset plus an additional header row.

**5. Final cancer predictions with stacked meta-classifier ensemble**

The individual cancer predictions and additional features are fed into the final meta-classifier ensemble:

```sh
# cwd: predict/
# --dataset may be {sample, stage1, stage2}
# --submission may be {1, 2}
nohup python -u 05_cancer_pred_meta_ens.py --dataset stage2 --submission 1 > 05_cancer_pred_meta_ens.out.log 2>05_cancer_pred_meta_ens.err.log &
```

## Training process

**1. Preprocess DICOM studies**

`01_preprocess.py`

See step 1 in "Predict" section.

**2. Create volumes for sex determination module**

`02_create_volumes_sex_determination.py`

Creates downsampled (32, 32, 64) 3-D volumes together with manually labeled patient sex info.

**2b. Train sex determination module**

Train models in sex_det:

```sh
# cwd: train/sex_det/
CUDA_VISIBLE_DEVICES=0 nohup python -u sd01a.py > sd01a.out.log 2>sd01a.err.log &
```

**3. Create annotations data**

`03_create_annotations.py`

The output pickle files containing nodule annotations data by study is already provided in the annotations folder. This script simply creates these pickle files from the annotation coordinates data file.

**4. Create training data for cancer predictions from nodule annotations**

`04_create_patches_cancer_pred_anno.py`

Many configurations were tried, and ultimately our best models were trained based on the following:

dimensions = '2d'
patchsize = 32
scaling = 'stretch'
multiple = 'largest'
padding = 0
offcenter = 25
rotation = 0
view = 'axial'

These are specified to the script as flags, i.e. `--dimensions 2d --patchsize 32 ...`.

**4b. Train cancer prediction on nodules**

Train models in cancer_pred_anno:

```sh
# cwd: train/cancer_pred_anno/
CUDA_VISIBLE_DEVICES=0 nohup python -u resnet2d09d.py > resnet2d09d.out.log 2>resnet2d09d.err.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u resnet2d09e.py > resnet2d09e.out.log 2>resnet2d09e.err.log &
CUDA_VISIBLE_DEVICES=2 nohup python -u resnet2d09f.py > resnet2d09f.out.log 2>resnet2d09f.err.log &
```

**5. Create training data for nodule detection modules**

`05_create_patches_nodule_detect.py`

Numerous configurations were tried, but our ultimate included models were trained based on the following configurations passed as flags:
`--dimensions 2daxial3stack --patchsize 64 --iternum 1`
`--dimensions 2daxial5stack --patchsize 64 --iternum 1`
`--dimensions 3d --patchsize 64 --iternum 1`

**5b. Train nodule detection modules**

Train models in nodule_detect:

```sh
# cwd: train/nodule_detect/
CUDA_VISIBLE_DEVICES=0 nohup python -u m05a.py > m05a.out.log 2>m05a.err.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u m09a.py > m09a.out.log 2>m09a.err.log &
CUDA_VISIBLE_DEVICES=2 nohup python -u m10a.py > m10a.out.log 2>m10a.err.log &
```

**6. Create training data for nodule bounding box modules**

`06_create_patches_bbox.py`

We create two sets of training data, with the following passed flags:
`--nstack 1`
`--nstack 2`

**6b. Train nodule bounding box modules**

Train models in nodule_bbox:

```sh
# cwd: train/nodule_bbox/
CUDA_VISIBLE_DEVICES=0 nohup python -u m02a.py > m02a.out.log 2>m02a.err.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u m04a.py > m04a.out.log 2>m04a.err.log &
```

**7. Train stacked meta-classifier cross-validated ensemble**

`07_cancer_pred_meta_cv.py`

This is a stacked meta-classifier which is trained after the "predict" steps 1-4. Individual predictions `../predictions/preds_{1-750}.csv` and features at `../data_proc/stage1/features/config_{1-750}_features.csv` must be available. The number of folds for cross-validation is specified with `--folds 4` (defaults to 4).

For stage 2, we will change this script to utilize all the stage 1 training set + test set labels and retrain to create an additional new meta classifier model.


## License

Apache 2.0
