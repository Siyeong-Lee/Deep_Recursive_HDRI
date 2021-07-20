# Deep Recursive HDRI in Pytorch
[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Siyeong_Lee_Deep_Recursive_HDRI_ECCV_2018_paper.pdf)

We provide PyTorch implementations for GAN-based mutliple exposure stack generation.
- [x] Deep recursive HDRI

## General
If you use the code for your research work, please cite our papers.

```
@inproceedings{lee2018deep,
  title={Deep recursive hdri: Inverse tone mapping using generative adversarial networks},
  author={Lee, Siyeong and Hwan An, Gwon and Kang, Suk-Ju},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={596--611},
  year={2018}
}
```

### Model inference
* Conda environment
```
conda create -n hdr python=3.6
conda activate hdr
conda install -c anaconda mkl
conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch
```

* install requirements.txt
  * to install torch == 0.3.1 , 
```
pip install -r requirements.txt
```

* Please download two model weights below and organize the downloaded files as follows:
```
DeepRecursive_HDRI
├──Result
    └──model
       ├── HDRGAN_stopdown_G_param_ch3_batch1_epoch20_lr0.0002.pkl
       └── HDRGAN_stopup_G_param_ch3_batch1_epoch20_lr0.0002.pkl
```

* Prepare your test images
```
DeepRecursive_HDRI
├──input
   ├── test files (*.png etc..)
```

* 

### Model weight
| Model Name | model weight |
|:-------------------:|:------------:|
|Deep Recursive HDRI  | [stopdown](https://drive.google.com/file/d/1EBNzkpPAlb01baNhw878BTGkmQpjFKdJ/view?usp=sharing) <br> [stopup](https://drive.google.com/file/d/1qiCfOxOn7rfEbNrOvkp1RkpFk91hmvF3/view?usp=sharing) |

## License

Copyright (c) 2020, Siyeong Lee.
All rights reserved.

The code is distributed under a BSD license. See `LICENSE` for information.
