**Note: You can compile darknet using [`CMake`](https://github.com/AlexeyAB/darknet/blob/master/README.md#how-to-compile-on-linuxmacos-using-cmake), [`Powershell`](https://github.com/AlexeyAB/darknet/blob/master/README.md#using-also-powershell) and [`make`](https://github.com/AlexeyAB/darknet/blob/master/README.md#how-to-compile-on-linux-using-make). This guide is based on make (Default method for linux)**

# Linux Requirements

### GPU

*   [GPU with CC >= 3.0](https://developer.nvidia.com/cuda-gpus)
*   [Cuda](https://developer.nvidia.com/cuda-downloads?target_os=Linux) (Large File) [Post Installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
*   [OpenCV](https://docs.opencv.org/4.5.3/d7/d9f/tutorial_linux_install.html) (Be Patient...)
    * `sudo apt install libopencv-dev`
*   [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar)[ (Ubuntu)](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install)

### CPU

*   [OpenCV](https://docs.opencv.org/4.5.3/d7/d9f/tutorial_linux_install.html)
    * `sudo apt install libopencv-dev`

# Compile (CPU)
**For GPU we need to change some variables in [`Makefile`](https://github.com/AlexeyAB/darknet/blob/master/README.md#how-to-compile-on-linux-using-make)**

`git clone powershell https://github.com/AlexeyAB/darknet`

### Make
Set `OPENCV=1` in Makefile

`make`
