# Test
- Download models from [OneDrive](https://mailccsf-my.sharepoint.com/:f:/g/personal/zhipeng_mail_ccsf_edu/EkWlTFNurBZOh9s37U4BMWoBcQmxvyWPjqjJpuZ0O-cNTg?e=RtBJOX), [GoogleDrive](https://drive.google.com/drive/folders/19dBWxOqZnvM0FsgXGzH2Y7Bg7wgYMEoO?usp=sharing) or [BaiduDrive](https://pan.baidu.com/s/14_4XVoes3IZCe8xt1-GNQg), and put them to `snapshot` directory

- BaiduDrive extracted code `uqvi`

## Test on a specific video
eg,
```
python siamese_tracking/run_video.py --arch SiamRPNRes22 --resume snapshot/CIResNet22_RPN.pth --video videos/bag.mp4
```
- The opencv version here is 4.1.0.25, and older versions may be not friendly to some functions.
- If you try to conduct this project on a specific tracking task, eg. pedestrian tracking, it's suggested that you can tuning hyper-parameters on your collected data with our tuning toolkit detailed below. 

## Test through webcam
eg,
```
python siamese_tracking/run_webcam.py --arch SiamRPNRes22 --resume snapshot/CIResNet22_RPN.pth
```
- The opencv version here is 4.1.0.25, and older versions may be not friendly to some functions.
- You can embed any tracker for fun. This is also a good way to design experiments to determine how environmental factors affect your tracker.


## Test on benchmarks

### Data preparation

The test dataset (OTB or VOT) should be arranged in `dataset` directory. Your directory tree should look like this:
```
${Tracking_ROOT}
|—— experimnets
|—— lib
|—— snapshot
|—— dataset
  |—— OTB2013.json
  |—— OTB2015.json
  |—— VOT2015
     | —— videos...
  |—— OTB2013 (or VOT2015...)
     |—— videos...
|—— run_tracker.py
|—— ...

```
[OTB2013.json](https://drive.google.com/file/d/1ZV6m2cN_TnM8XKR0q3ElYEz0P23iy2qn/view?usp=sharing) and [OTB2015.json](https://drive.google.com/file/d/1eIq7pCz_ik2toO1l9Npk1WXk4mZPK9_N/view?usp=sharing) can be download here.


### Conda preparation
```
sh install_fc.sh or sh install_rpn.sh
```

### Toolkit preparation
- Set up vot-toolkit according to official [tutorial](http://www.votchallenge.net/howto/integration_channels.html)
- Modify `path_to/toolkit` in `lib/core/get_eao.m` to your vot-toolkit path
- In your matlab install path (MATLAB2017b or higher),
```
cd $matlab_path/R2018b/extern/engines/python
python setup.py install 
```


### Run tracker


```bash
CUDA_VISIBLE_DEVICES=0 python ./siamese_tracking/test_siamfc.py --arch SiamFCRes22 --resume ./snapshot/CIResNet22.pth --dataset OTB2013
or 
CUDA_VISIBLE_DEVICES=0 python ./siamese_tracking/test_siamrpn.py --arch SiamRPNRes22 --resume ./snapshot/CIResNet22_RPN.pth --dataset VOT2017 --cls_type thinner
```


### Analysz testing results
- OTB
```bash
python ./lib/core/eval_otb.py OTB2013 ./result SiamFC* 0 1
```

- VOT
```bash
python ./lib/core/eval_vot.py VOT2017 ./result 2>&1 | tee logs/siamfc_eval.log
```

### Attention !!
- Recently we found that the image is slightly inconsistent while using different OpenCV version. And the speed of some opencv versions are relatively slow for some reason. It is recommended that you install packages above. 
- The SiamRPN based model is trained on pytorch0.4.1, since we found that memory leak happens while testing SiamRPN on pytorch0.3.1 with multithread tools.


<!-- <div id="VOT-TEST"></div>

#### **For VOT**
1) Please refer to VOT official [tutorial](http://www.votchallenge.net/howto/workspace.html) to set up your workspace. <br/>
2) Move `txt` result files to `result` directory in vot-workspace. Please keep directory name coincident with `run_analysis.m`. <br/>
3) run `run_analysis.m` -->
