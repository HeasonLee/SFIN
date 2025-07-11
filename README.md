# SFIN
Code and data of CVPR 2025 paper "Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement". [Paper PDF](https://arxiv.org/pdf/2504.02555) 

<p align="center">
  <img width="900" src="git.png">
</p>

# How to Use
1. Down checkpoints [here](https://pan.baidu.com/s/1mOMZGUwRHxZpbYvrtYb30g?pwd=dgf5). Files with `_bf` are for BF mode and others are for HAADF mode.
2. Unzip test dataset `haadf_data_test` and `bf_data_test.zip`.
3. Run `ours_gpu_demo.py` to get enhancement/detection results.
4. Run `metrics.py` to get PSNR/SSIM metrics.

If you have any other needs, please contact us via email [lihesong2@bit.edu.cn](lihesong2@bit.edu.cn) or my WeChat: linsfriend.


# Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).

```
@article{Li2025SFIN,
    title = {Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement},
    author = {Li, Hesong and Wu, Ziqi and Shao, Ruiwen and Zhang, Tao and Fu, Ying},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2025},
}
