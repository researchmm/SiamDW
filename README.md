# Deeper and Wider Siamese Networks for Real-Time Visual Tracking
we are hiring talented interns: houwen.peng@microsoft.com
## News
- :trophy: **We are the Winner of [VOT-19 RGB-D](http://data.votchallenge.net/vot2019/vot2019_rgbt.pdf) challenge**
- :trophy: **We won the Runner-ups in VOT-19 Long-term and RGB-T challenges**
- :sunny::sunny: We add the results on **VOT-18**, **VOT-19**, **GOT10K**, **VISDRONE19**, and **LaSOT** datasets.
- :sunny::sunny: The training and testing code of SiamFC+ and SiamRPN+ have been released.
- :sunny::sunny: Our [paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Deeper_and_Wider_Siamese_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.html) has been accepted by [CVPR2019](http://openaccess.thecvf.com/menu.py) (**Oral**).
- :sunny::sunny: We provide a [parameter tuning toolkit](#TUNE-TOOLKIT) for siamese tracking framework.


## Introduction
Siamese networks have drawn great attention in visual tracking because of their balanced accuracy and speed.  However, the backbone network utilized in these trackers is still the classical AlexNet, which does not fully take advantage of the capability of modern deep neural networks. 
  
Our proposals improve the performances of fully convolutional siamese trackers by,
1) introducing CIR and CIR-D units to unveil the power of deeper and wider networks like [ResNet](https://arxiv.org/abs/1512.03385) and [Inceptipon](https://arxiv.org/abs/1409.4842); 
2) designing backbone networks according to the analysis on internal network factors (e.g. receptive field, stride, output feature size), which affect tracking performances.

<div align="center">
  <img src="demo/vis.gif" width="800px" />
  <!-- <p>Example SiamFC, SiamRPN and SiamMask outputs.</p> -->
</div>

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
| CIResNet22-RPN| 0.674 | 0.666 | 0.381 | 0.376 | 0.294 |

#### Main results trained with GOT-10k (SiamFC)
| Models  | OTB13 | OTB15 | VOT15 | VOT16 | VOT17|
| :------ | :------: | :------: | :------: | :------: | :------: |
| Alex-FC        |-      | -     | -     | -     |0.188     | 
| CIResNet22-FC  | 0.664 | 0.654 | 0.361 | 0.335 | 0.266| 
| CIResNet22W-FC | **0.689** | **0.674** | **0.368** | **0.352** | **0.269** |
| CIResIncep22-FC| 0.673 | 0.650 | 0.332 | 0.305 | 0.251|
| CIResNext22-FC | 0.668 | 0.651 | 0.336 | 0.304 | 0.246|
| Raw Results | :paperclip: [OTB2013](https://pan.baidu.com/s/1HgkjUmnYl7qagIkz9u4r_A) | :paperclip: [OTB2015](https://pan.baidu.com/s/1ZgL4DQL57cuWfqxLFmUR1A)  | :paperclip: [VOT15](https://pan.baidu.com/s/1SGLcMWgrBuBT_kaXMdQBug)  | :paperclip: [VOT16](https://pan.baidu.com/s/12jmWEwo4tjbM4SHSKgULNw) |  :paperclip: [VOT17](https://pan.baidu.com/s/1UWQRE2VrJrONpj293el4Pw) |

- Some reproduced results listed above are slightly better than the ones in the paper.
- Recently we found that training on GOT10K dataset can achieve better performance for SiamFC. So we provide the results being trained on GOT10K.
- Download pretrained on GOT10K [model](https://drive.google.com/file/d/1xvexXCUCB0gCYFnShj3NQ4Xuk52lLLtE/view?usp=sharing). 


#### New added results
| Benchmark | VOT18| VOT19 | GOT10K | VISDRONE19 | LaSOT | 
|:------: |:------: | :------: |  :------: | :------: | :------: | 
| Performance   | **0.270** | **0.242** | **0.416**  | **0.383** |**0.387**|
| Raw Results | :paperclip: [VOT18](https://pan.baidu.com/s/1hKg-n4PTPL_VCEdxCrXMAA) | :paperclip: [VOT19](https://pan.baidu.com/s/1mwPrMJhi79_TO40RzTAwvQ) | :paperclip: [GOT10K](https://pan.baidu.com/s/10INTbmtfL-EdfkAmDQgcKw) |:paperclip: [VISDRONE](https://pan.baidu.com/s/17MLGaHEFEFG3yWUmLqJ7ig) | :paperclip: [LaSOT](https://pan.baidu.com/s/1SdQVcUsJG0TeGI77H92_UQ) |

- We add resutls of SiamFCRes22W on recent benchmarks.  

#### Environment
The code is developed with Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz GPU: NVIDIA .GTX1080



## Quick Start
### Test
See details in [test.md](lib/tutorials/test.md)

### Train
See details in [train.md](lib/tutorials/train.md)

:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:

## Citation
If any part of our paper and code is helpful to your work, please generously cite with:

```
@InProceedings{SiamDW_2019_CVPR,
author = {Zhang, Zhipeng and Peng, Houwen},
title = {Deeper and Wider Siamese Networks for Real-Time Visual Tracking},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
} 
```

## License
Licensed under an MIT license.



