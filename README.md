# DreamArrangement: Language-Conditioned Robotic Object Rearrangement

This repository holds the PyTorch implementation for [DreamArrangement: Learning Language-conditioned Robotic Rearrangement of Objects via Denoising Diffusion and VLM Planner](https://wenkai-chen.com/publication/dreamarrangement).


## Environment
Install the conda environment:
```
conda create -n dreamrearrange python=3.8
conda install pytorch=1.11.0 cudatoolkit=11.3 torchvision=0.12.0 -c pytorch
pip install opencv-python scipy scikit-learn matplotlib pandas ortools omegaconf ipykernel ipywidgets
pip install git+https://hub.nuaa.cf/openai/CLIP.git
```


## Datasets

* `2D-Tabletop dataset`: It includes two parts: [YCB_kitchen_data](https://drive.google.com/file/d/1FI0XiT3d7KeG4ScIXyHBdu_mqrV3mSVi/view?usp=drive_link) for horizontal, vertical, and circle scenes, [YCB_Inpainted_data](https://drive.google.com/file/d/1Y_6Te50msNpeA6TrClUOmItbO9PwUHL-/view?usp=drive_link) for cantaining scenes. The data are already processed and can be directly used for training. You should put them in the `data` folder.

<img src="./pics/dataset.jpg" alt= “” width="600" height="value" style="vertical-align:middle;margin:0px 40px">


## Training
### 2D-Tabletop
To train for 2D-Tabletop data:
```
python train.py  --train 1 --use_position 0 --use_time 0 --train_epoch 30000  --train_pos_noise_level_stddev 1.0 --train_ang_noise_level_stddev 1.047198  --train_within_floorplan 1 --train_batch_size 64 --text_form word --data_augment 4 --use_emd 1 --use_move_less 1 --data_type YCB_kitchen YCB_Inpainted
```

### Pretrained Weights
We provide DreamArrangement weights pretrained on 2D-Tabletop for 30k iterations.
* [best-model weight](https://drive.google.com/file/d/1FrwXlp-LRbcMn8wuJqYLTei06BxmwsSG/view?usp=drive_link)


## Evaluation
### 2D-Tabletop
To run evaluation for 2D-Tabletop data:
```
python train.py  --train 0 --use_position 0 --use_time 0 --denoise_within_floorplan 1 --text_form word --use_emd 1 --use_move_less 1 --data_type YCB_kitchen YCB_Inpainted --model_path <full-path-to-model>
```

<!-- Ground Truth            |  Initial |  Denoised
:-------------------------:|:-------------------------: |:-------------------------:
<img src="./README_media/inference/14_groundtruth.jpg" alt= “” width="260" height="value" style="vertical-align:middle;margin:0px 0px">  |  <img src="./README_media/inference/14_initial.jpg" alt= “” width="260" height="value" style="vertical-align:middle;margin:0px 0px"> | <img src="./README_media/inference/14_trans50000-grad_nonoise.jpg" alt= “” width="260" height="value" style="vertical-align:middle;margin:0px 0px"> -->


## Acknowledgments
This code repository is heavily based on [LEGO-Net](https://github.com/QiuhongAnnaWei/LEGO-Net).


## Citation
If you find our work useful in your research, please cite: