# A  Novel Window-Interaction Module Based on W-MSA

## Project for Statistical Learning Course
<!-- [Ruochen Cui](https://github.com/421zuoduan)  [Mingjun Ni](https://github.com/TsukiRinA) -->

> **Abstract:** *W-MSA proposed by Swin Transformer has limitations in facilitating information interaction between windows. To address this, we introduce a module that utilizes convolution to achieve inter-window information interaction across different regions. Experiments demonstrate that our proposed module, when combined with W-MSA in a dual-branch structure, outperforms the simple W-MSA. In the deraining task conducted on the Uformer, we observe a 0.14dB improvement in performance. Our code can be found at https://github.com/421zuoduan/WIM-code.* 
<hr />


This repository contains the code used to conduct the experiments for our ICLR 2024 Tiny Paper submission for the paper titled "Window Interaction Within Swin Transformer".


## Our Work

The final version can be found in `models/compared_trans/Uformer_KAv4/model.py`. Following codes are for ablation study.

* Uformer_KAv7：without window convolutions, only global convolution added
* Uformer_KAv9: based on KAv7, remove the shift window in W-MSA
* Uformer_KAv10：based on KAv4，remove the operation of Self Attention and SE module
* Uformer_KAv11：based on KAv4, remove the SE module for global convolution kernel
* Uformer_KAv12：based on KAv4, remove the shift window in W-MSA


## Dataset

The dataset employed in this study is Rain100L, which comprises 200 image pairs in the training set and 100 image pairs in the testing set. Unfortunately, the dataset originally provided at the disclosed URL has been replaced with Rain200L. To ensure the dataset is in the correct structure, `dataset_process.py` can be utilized for preprocessing.

<table>
  <tr>
    <th align="left">Derain</th>
    <th align="center">Dataset</th>
  </tr>
  <tr>
    <td align="left">Rain100L</td>
    <td align="center"><a href="https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html">Link</a></td>
  </tr>
</table>


## Training Setting

The learning rate is typically set to 1e-4, and the AdamW optimizer is employed with a random seed set to 1. The input image size for the model is 128*128, and both L1 loss and SSIM loss are utilized as loss functions. We have trained models for 1000 epochs. Details can be found in `configs/option_model.py`


## Training and Evaluation

- Training. Revise and run `python run_derain.py` for training
- Evaluation. Revise and run `python run_derain_test.py` for testing.

Details can be found [here](https://github.com/XiaoXiao-Woo/derain).



## Results

### Evaluation of Models

|Model|PSNR|SSIM|Params|FLOPs|
|-|-|-|-|-|
|Uformer|38.680|0.9788|20.628M|10.308G|
|Ours|**38.820**|**0.9795**|24.667M|13.548G|



**Note**

* Download datasets and put it with the following format. 

* We introduce more methods based on Restormer to the single image deraing task.

* The project is based on MMCV, but you needn't to install it and master MMCV. More importantly, it can be more easy to introduce more methods.

* We have modified part of the code for reading dataset, specifally for Rain100L, which can be read as the same way with Rain200L. But you have to operate on the dataset. Target and rainy figure should be operated into one figure. You can finish the operation above through dataset_process.py, *but be careful to use them*.


```
|-$ROOT/datasets
├── Rain100L
│   ├── train_c
│   │   ├── norain-001.png
│   │   ├── ...
│   ├── test_c
│   │   │   ├── norain-001.png
│   │   │   ├── ...
```
