# Darknet Configurations
**This documentation is for Google Colab. If you want to know how to compile darknet on your linux local machine (Ubuntu 20.04), please read [this documentation](https://github.com/amirafshari/LPD-YOLOv4/blob/master/darknet-linux.md).**

```python
# clone repo
#!git clone https://github.com/AlexeyAB/darknet
!git clone https://github.com/amirafshari/LPD-YOLOv4
```

## GPU


```python
# change makefile to have GPU and OPENCV enabled
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```


```python
# verify CUDA
!/usr/local/cuda/bin/nvcc --version
```


```python
# make darknet
!make
```

## Weights


```python
# pre-trained weights on MS COCO dataset
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```


```python
# pre-trained weights for the convolutional layers
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

## Generate train.txt and test.txt
These files are not in the official repo, but you can find them in my repository.


```python
!python generate_train.py
!python generate_test.py
```

## Configurations
We need to change/create these files (I configured them for our object (which is license plate), and put them in this repository):
*   data/obj.names
*   data/obj.data
*   cfg/yolov4-custom.cgf
*   cfg/yolov4-obj.cfg


# Training

## Configurations
https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects  

*   1 Epoch = images_in_train_txt / batch = 2000 / 32 = 62.5



## Train


```python
# Access Denied Error
!chmod +x ./darknet
```


```python
# set custom cfg to train mode 
%cd cfg
!sed -i 's/batch=1/batch=64/' yolov4-obj.cfg
!sed -i 's/subdivisions=1/subdivisions=16/' yolov4-obj.cfg
%cd ..
```


```python
!./darknet detector train ./data/obj.data ./cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show -map
```

## Restart
In case of intruption, we can restart training from our last weight.  
(every 100 iterations our weights are saved to backup folder in yolov4-obj_last.weights) (~every 30 minutes)  
(every 1000 iterations our weight are saved to backup folder in yolo-obj_xxxx.weights)


```python
!./darknet detector train ./data/obj.data ./cfg/yolov4-obj.cfg ./backup/yolov4-obj_last.weights -dont_show -map
```

# Sanity Check

#### Setup

```python
# set custom cfg to test mode 
%cd cfg
!sed -i 's/batch=64/batch=1/' yolov4-obj.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' yolov4-obj.cfg
%cd ..
```


```python
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
```

#### COCO Dataset

```python
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg
```

```python
imShow('./predictions.jpg')
```
![download](https://user-images.githubusercontent.com/17769927/138662797-827178bd-ce03-4896-b093-1705c3ac6d4f.png)

#### Custom Dataset
```python
!./darknet detector test ./data/obj.data ./cfg/yolov4-obj.cfg ./backup/yolov4-obj_last.weights ../Cars354.png -thresh 0.3
```

```python
imShow('./predictions.jpg')
```
![result-4](https://user-images.githubusercontent.com/17769927/134551901-37ff3f6d-37ae-42dc-96c3-8064786355fe.jpg)


**To process a list of images data/train.txt and save results of detection to result.json file use**

```python
!./darknet detector test data/obj.data cfg/yolov4-obj.cfg backup/yolov4-obj_last.weights -ext_output -dont_show -out result.json < data/test.txt
```

# Metrics
**Use -map flag while training for charts**  
mAP-chart (red-line) and Loss-chart (blue-line) will be saved in root directory.  
mAP will be calculated for each 4 Epochs ~ 240 batches

```python
!./darknet detector map data/obj.data cfg/yolov4-obj.cfg backup/custom.weights
```

