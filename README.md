# Visual-Selective-VIO (ECCV 2022)

This is the test code for the paper "Efficient Deep Visual and Inertial Odometry with Adaptive Visual Modality Selection". 

![Structure](figures/figure.png)  

## Data Preparation

The code in this repository is tested on KITTI Odometry dataset. To download the KITTI dataset, please run `data/data_prep.sh`. The IMU data after pre-processing is provided under `data/imus`. 

After downloading the dataset, run `preprocess.py` to generate relative poses.

## IMU data format

The IMU data has 6 dimentions: 

1. acceleration in x, i.e. in direction of vehicle front (m/s^2)
2. acceleration in y, i.e. in direction of vehicle left (m/s^2)
3. acceleration in z, i.e. in direction of vehicle top (m/s^2)
4. angular rate around x (rad/s)
5. angular rate around y (rad/s)
6. angular rate around z (rad/s)

## Download pretrainined models

Two pretrained models `vf_512_if_256_3e-05.model` and `vf_512_if_256_5e-05.model` are provided in [Link](https://drive.google.com/drive/folders/1KrxpvUV9Bn5SwUlrDKe76T2dqF1ooZyk). Please download the models and place them under `models` directory.

## Test the model

Example command:

      python test.py --gpu_ids '0' --model_name 'vf_512_if_256_3e-05'  

The figures and error records will be generated under `results`

## Results example

Estimated path (left), speed heatmap (middle) and decision heatmap (right) for path 07 using `vf_512_if_256_5e-05.model`. 

<img src="figures/07_path_2d.png" alt="path" height="230"/> <img src="figures/07_decision_smoothed.png" alt="path" height="230"/> <img src="figures/07_speed.png" alt="path" height="230"/>

## Reference

> Mingyu Yang, Yu Chen, Hun-Seok Kim, "Efficient Deep Visual and Inertial Odometry with Adaptive Visual Modality Selection"

    @article{yang2022efficient,
      title={Efficient Deep Visual and Inertial Odometry with Adaptive Visual Modality Selection},
      author={Yang, Mingyu and Chen, Yu and Kim, Hun-Seok},
      journal={arXiv preprint arXiv:2205.06187},
      year={2022}
    }
