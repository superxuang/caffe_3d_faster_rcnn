# Caffe with 3D Faster R-CNN
This is a modified version of [Caffe](https://github.com/BVLC/caffe) which supports the **3D Faster R-CNN framework** and **3D Region Proposal Network** as described in our paper [**Efficient Multiple Organ Localization in CT Image using 3D Region Proposal Network**]([Early access](http://doi.org/10.1109/TMI.2019.2894854)).

This code has been compiled and passed on `Windows 7 (64 bits)` using `Visual Studio 2013`.

## How to build

**Requirements**: `Visual Studio 2013`, `ITK-4.10`, `CUDA 8.0` and `cuDNN v5`

### Pre-Build Steps
Please make sure CUDA and cuDNN have been installed correctly on your computer.

Clone the project by running:
```
git clone https://github.com/superxuang/caffe_3d_faster_rcnn.git
```

In `.\windows\Caffe.bat` set `ITK_PATH` to ITK intall path (the path containing ITK `include`,`lib` folders).

### Build
Run `.\windows\Caffe.bat` and build the project `caffe` in `Visual Studio 2013`.

## License and Citation

Please cite our paper and Caffe if it is useful for your research:

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }