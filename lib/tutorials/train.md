# Train

This is a generic framework for siamese tracking. Follow the instructions to embed your ideas to this code easily.

## Data preparation
- We provide pre-processed `VID`, `YTB`, `GOT10K`, `COCO`, `DET` and `LASOT`. You can download it from [GoogleDrive](https://drive.google.com/drive/folders/1uknmCp927CcpNRoqiC7L4M-QIh60dQ2p?usp=sharing) or [BaiduDrive](https://pan.baidu.com/s/17_qGxspaXC96SzyPwP5rmQ).
- BaiduDrive extracted code `bnd9`
- Some data can only be downloaded from GoogleDrive since uploading size limitation of BaiduDrive.

## Pretrained model preparation
Our code will download pretrained model from GoogleDrive automatically. If failed, please download from [OneDrive](https://mailccsf-my.sharepoint.com/:u:/g/personal/zhipeng_mail_ccsf_edu/EXLC8YnM9B9Kq5KcqfjbFg4B-OIwp6ZflvW_p0s0K3R1_Q?e=XNqj3n), [GoogleDrive](https://drive.google.com/open?id=1RIMB9542xXp60bZwndTvmIt2jogxAIX3) or [BaiduDrive](https://pan.baidu.com/s/10VnODB7alKuLezt49Vfb3Q), and put them to `pretrain` directory.


## Conda preparation
```
sh install_fc.sh or sh install_rpn.sh
```

## Toolkit preparation
Modify `path_to/toolkit` in `lib/core/get_eao.m` to your vot-toolkit path.

## Setting preparation
Modify yaml files in `experiment/train/` according to your needs.

## One-key Running
This script will excute `train`-`epoch-test`-`hyper-parameter tuning` automatically to save your time.
```
python siamese_tracking/onekey_fc.py
```
or 

```
python siamese_tracking/onekey.py
```

## Watch tuning process
The tuning toolkit will not stop unless you do. So we provide scripts to watch its process. 600-1000 groups and 1000-3000 groups for SiamFC and SiamRPN respectively is a good choice from my experience.

```
sh lib/utils/watch_tpe.sh or sh lib/utils/watch_tpe_fc.sh
```

# Implement your ideas on our code

## Data optimization
- We provide a [demo](https://drive.google.com/file/d/1xytp3_vcaFC9Hvlqvei1PL8P5MhI7f0Y/view?usp=sharing) to generate training pairs. You can modify it to process any data you want.
- Different training data and mix-up ratio will affect final performance. You can modify `WITCH_USE` in yaml files of `experiment/train/` to find witch data is better for your task. Also, modify `USE` in yaml files to try different mix-up ratio. High quality training data is beneficial to training. `GOT10K` and `LASOT` are recommended.

## Backbone optimization
We provide `ResNet`, `Inception` and `ResNext` in codes. However, there are many other choices like `DenseNet`, `NasNet`. I got an email from a reader who said that DenseNet can get better results in my framework. <br/>
Add your backbone in `lib/models/backbone.py`. Pretraining backbone on Imagenet is always good to training.

## Cross-correlation optimization
Cross-correlation is the key operation in both SiamFC and SiamRPN. There may be optima way to replace it. For example, `cosine distance` and `normalized L2 distance` is better for SiamFC from my experience. <br/>

Add your strategy in `lib/models/connect.py`.

## Loss optimization
We use the simplest loss function in training. You can try more optimal loss function like `focal loss`. <br/>

Add your loss function in `lib/models/siamfc.py` or `lib/models/siamrpn.py`

<br/>


===============================================================================
Further discussion or questions about code please email `zhangzhipeng2017@ia.ac.cn`. <br/>

Let's learn to progress together.
