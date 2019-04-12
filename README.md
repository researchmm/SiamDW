# Deeper and Wider Siamese Networks for Real-Time Visual Tracking
we are hiring talented interns: houwen.peng@microsoft.com
## News
- :sunny::sunny: Our [paper](https://arxiv.org/abs/1901.01660) have been accepted by [CVPR2019](http://openaccess.thecvf.com/menu.py) (**Oral**).
- :sunny::sunny: We provide a [parameter tuning toolkit](#TUNE-TOOLKIT) for siamese tracking framework.


## Introduction
Siamese networks have drawn great attention in visual tracking because of their balanced accuracy and speed.  However, the backbone network utilized in these trackers is still the classical AlexNet, which does not fully take advantage of the capability of modern deep neural networks. 
  
Our proposals improve the performances of fully convolutional siamese trackers by,
1) introducing CIR and CIR-D units to unveil the power of deeper and wider networks like [ResNet](https://arxiv.org/abs/1512.03385) and [Inceptipon](https://arxiv.org/abs/1409.4842); 
2) designing backbone networks according to the analysis on internal network factors (e.g. receptive field, stride, output feature size), which affect tracking performances.

<!-- :tada::tada: **Highlight !!**
Siamese tracker is severely sensitive to hyper-parameter, which is a common sense in tracking field. Although significant progresses have been made in some works, the result is hard to reproduce. In this case, we provide a [parameter tuning toolkit]() to make our model being reproduced easily. We hope our efforts and supplies will be helpful to your work. -->

## Main Results
#### Main results on VOT and OTB
| Models  | OTB13 | OTB15 | VOT15 | VOT16 | VOT17| 
| :------ | :------: | :------: | :------: | :------: | :------: |
| Alex-FC      | 0.608 | 0.579 | 0.289 | 0.235 | 0.188 | 
| Alex-RPN     | -     | 0.637 | 0.349 | 0.344 | 0.244 |
| CIResNet22-FC  | 0.663 | 0.644 | 0.318 | 0.303 | 0.234 |
| CIResIncep22-FC| 0.662 | 0.642 | 0.310 | 0.295 | 0.236 |
| CIResNext23-FC | 0.659 | 0.633 | 0.297 | 0.278 | 0.229 |
| CIResNext22-RPN| 0.674 | 0.666 | 0.381 | 0.376 | 0.294 |

#### Main results training with GOT-10k (SiamFC)
| Models  | OTB13 | OTB15 | VOT15 | VOT16 | VOT17| 
| :------ | :------: | :------: | :------: | :------: | :------: |
| CIResNet22-FC  | 0.664 | 0.654 | 0.361 | 0.335 | 0.266|  
| CIResNet22W-FC | **0.689** | **0.664** | **0.368** | **0.352** | **0.269** |  
| CIResIncep22-FC| 0.673 | 0.650 | 0.332 | 0.305 | 0.251|  
| CIResNext22-FC | 0.668 | 0.651 | 0.336 | 0.304 | 0.246|  

- Some reproduced results listed above are slightly better than the ones in the paper.
- Recently we found that training on GOT10K dataset can achieve better performance for SiamFC. So we provide the results being trained on GOT10K.
- CIResNet22W-FC is our recent work, which is not included in our paper.

#### Note
- You can download raw results from [GoogleDrive](https://drive.google.com/file/d/1rTC2XKJ2bznVjtXW-UAzeUGc7QizeLP9/view?usp=sharing), [OneDrive](https://mailccsf-my.sharepoint.com/:f:/g/personal/zhipeng_mail_ccsf_edu/Ekjf2LfnGJ9NkYladR_Uk3IBnIQ3HlQybjzFRkwgeetGqg?e=DLlPJO) and [BaiduDrive](https://pan.baidu.com/s/1J1x58GaKtbMISDVv0ZuoCg) without running the code.
- Extracted code for Baidu Drive is `htyx`


#### Environment
The code is developed with Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz GPU: NVIDIA .GTX1080

:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:

## Quick Start
### Installation
**For SiamFC**
```
conda create -n SiamDWFC python=3.6
source activate SiamDWFC
sh install_fc.sh
```
**For SiamRPN**
```
conda create -n SiamDWRPN python=3.6
source activate SiamDWRPN
sh install_rpn.sh
```

- Recently we found that the image is slightly inconsistent while using different OpenCV version. And the speed of some opencv versions are relatively slow for some reason. It is recommended that you install packages above. 
- The SiamRPN based model is trained on pytorch0.4.1, since we found that memory leak happens while testing SiamRPN on pytorch0.3.1 with multithread tools.

### Data preparation
**For testing** <br/>
The test dataset (OTB or VOT) should be arranged in `dataset` directory. Your directory tree should look like this:
```
${Tracking_ROOT}
|—— experimnets
|—— lib
|—— snapshot
|—— dataset
  |—— OTB2013.json
  |—— OTB2015.json 
  |—— OTB2013 (or VOT2015...)
     |—— videos...
|—— run_tracker.py
|—— ...

```
[OTB2013.json](https://drive.google.com/file/d/1ZV6m2cN_TnM8XKR0q3ElYEz0P23iy2qn/view?usp=sharing) and [OTB2015.json](https://drive.google.com/file/d/1eIq7pCz_ik2toO1l9Npk1WXk4mZPK9_N/view?usp=sharing) can be download here.

**For training SiamFC** <br/>
- We pre-process `VID` and `GOT10K` to training pairs. You can download it from [GoogleDrive](https://drive.google.com/file/d/1oNpN-oQq_L2bwZhYicS_JVVmC3hC9NT0/view?usp=sharing) or [BaiduDrive](https://pan.baidu.com/s/17_qGxspaXC96SzyPwP5rmQ).
- BaiduDrive extracted code `bnd9`
  
### Test
Download model from [OneDrive](https://mailccsf-my.sharepoint.com/:f:/g/personal/zhipeng_mail_ccsf_edu/EkWlTFNurBZOh9s37U4BMWoBcQmxvyWPjqjJpuZ0O-cNTg?e=RtBJOX), [GoogleDrive](https://drive.google.com/drive/folders/19dBWxOqZnvM0FsgXGzH2Y7Bg7wgYMEoO?usp=sharing) or [BaiduDrive](https://pan.baidu.com/s/14_4XVoes3IZCe8xt1-GNQg), and put them to `snapshot` directory
- BaiduDrive extracted code `uqvi`

```bash
CUDA_VISIBLE_DEVICES=0 python ./siamese_tracking/test_siamfc.py --arch SiamFCRes22 --resume ./snapshot/CIResNet22.pth --dataset OTB2013
or 
CUDA_VISIBLE_DEVICES=0 python ./siamese_tracking/test_siamrpn.py --arch SiamRPNRes22 --resume ./snapshot/CIResNet22_RPN.pth --dataset VOT2017
```
- Extracted code for Baidu drive is required due to softerware maintenance recently. Please input `v5du` in the download box. 

### Analysz testing results
#### **For OTB**
```bash
python ./lib/core/eval_otb.py OTB2013 ./result SiamFC* 0 1
or
python ./lib/core/eval_otb.py OTB2013 ./result SiamRPN* 0 1
```

<div id="VOT-TEST"></div>

#### **For VOT**
1) Please refer to VOT official [tutorial](http://www.votchallenge.net/howto/workspace.html) to set up your workspace. <br/>
2) Move `txt` result files to `result` directory in vot-workspace. Please keep directory name coincident with `run_analysis.m`. <br/>
3) run `run_analysis.m`

:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:
### Reproduce -- Train/Test/Tune

#### Preparation
- prepare conda environment and matlab-python API according to details above
- modify dataset path in  `experiments/train/*.yaml` to your needs.
- download pretrained model from [OneDrive](https://mailccsf-my.sharepoint.com/:u:/g/personal/zhipeng_mail_ccsf_edu/EXLC8YnM9B9Kq5KcqfjbFg4B-OIwp6ZflvW_p0s0K3R1_Q?e=XNqj3n), [GoogleDrive](https://drive.google.com/open?id=1RIMB9542xXp60bZwndTvmIt2jogxAIX3) or [BaiduDrive](https://pan.baidu.com/s/1TmIW8AsLEr9Mk3qSsT1pIg), and put them to `pretrain` directory
- Extracted code for Baidu drive is required due to softerware maintenance recently. Please input `7rfu` in the download box. 
#### SiamFC

##### [Epoch Train]()
```
python ./siamese_tracking/train_siamfc.py --cfg experiments/train/SiamFC.yaml --gpus 0,1,2,3 --workers 32 2>&1 | tee logs/siamfc_train.log
```

##### [Epoch Test]()
If you want to test multi-epochs after training,
```
mpiexec -n 16 python ./siamese_tracking/test_epochs.py --arch SiamFCRes22 --start_epoch 30 --end_epoch 50 --gpu_nums=4 --threads 16 --dataset OTB2013 2>&1 | tee logs/siamfc_epoch_test.log
```
```
python ./lib/core/eval_otb.py OTB2013 ./result SiamFC* 0 100 2>&1 | tee logs/siamfc_eval_epoch.log
```


##### [Param-Tune]()
<div id="TUNE-TOOLKIT"></div>
Siamese trackers are severely sensitive to hyper-parameters in common sense. We provide a toolkit for selecting optimal hyper-parameters on a benchmark (for SiamFC). Wish our efforts will be helpful to your work. Only OTB is supported.

```
mpiexec -n 16  python ./siamese_tracking/tune_gene.py --arch SiamFCRes22 --resume ./snapshot/CIResNet22.pth --dataset OTB2013 --gpu_nums 4 2>&1 | tee logs/gene_tune_fc.log
```


:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:
## Citation
If any part of our paper and code is helpful to your work, please generously cite with:

```
@inproceedings{SiamDW_2019_CVPR,
    author={Zhang, Zhipeng and Peng, Houwen},
    title={Deeper and Wider Siamese Networks for Real-Time Visual Tracking},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}
```

## License
Licensed under an MIT license.



