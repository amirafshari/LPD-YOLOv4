# Data

## Kaggle

This dataset countains 433 images with PASCAL VOC annotations in .xml files  
https://www.kaggle.com/andrewmvd/car-plate-detection

### PASCAL VOC to YOLO


```python
# [x_xenter, y_center, obj_width, obj_height]
# run this in root directory of your dataset

import os
import bs4 as BeautifulSoup

annotations = []
for e in os.listdir('./cars/annotations/'):
    with open(os.path.join('./cars/annotations/', e)) as f:
        soup = BeautifulSoup(f, 'lxml')
                
        x_min = int(soup.xmin.string)
        x_max = int(soup.xmax.string)
        y_min = int(soup.ymin.string)
        y_max = int(soup.ymax.string)
        
        img_width = int(soup.width.string)
        img_height = int(soup.height.string)
        
        obj_width = (x_max - x_min) / img_width
        obj_height = (y_max - y_min) / img_height
        
        x_center = (x_max + x_min) / (2*img_width)
        y_center = (y_max + y_min) / (2*img_height)
        
        item = [float('%.4f'%x_center), float('%.4f'%y_center), float('%.4f'%obj_width), float('%.4f'%obj_height)]
        
        annotations.append(item)

annotations = np.array(annotations)

# Save as .txt
for i, row in enumerate(annotations):
    np.savetxt('./annotations/Cars{}.txt'.format(i), [row], delimiter=' ', fmt='%.4g')
```

## Download from Google Open Image
There are two frameworks to download automaticaly:  
https://storage.googleapis.com/openimages/web/download.html

### 1. Fiftyone (recommended by Google)

https://voxel51.com/docs/fiftyone/

### 2. OID Toolkit (My Approach)


```python
# clone repo
!git clone https://github.com/EscVM/OIDv4_ToolKit
```


```python
%cd ./OIDv4_ToolKit
```

#### 1. Download and Split
*   Train set: 2000 Images 
*   Validation set: 386 Images 

*All available validation images are 386 images


```python
# Train set
!python3 main.py downloader --classes Vehicle_registration_plate --type_csv train --limit 400
```


```python
# Validation + Test ==> 20% of the size of train set
!python3 main.py downloader --classes Vehicle_registration_plate --type_csv validation --limit 400
```

#### 2. Convert to YOLOv4

##### 1. Change classes.txt (in root directory of OIDv4_ToolKit) to 'Vehicle registration plate'

##### 2. Run this code in OID Tookit root directory

*   This file doesnt exist in official repo but you can find it in my repository.




```python
!python convert_annotations.py
```

##### 3. Delete old labels
Delete 'label' folder in  

*   OID/Dataset/train 
*   OID/Dataset/validation


#### 3. Zip & Upload to Google Drive & mount drive

*   Train set as obj.zip
*   Validation set as test.zip




```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    

#### 4. Unzip and move to darknet/data

# EDA

*   Create a DataFrame from annotations to visualize our objects.
*   You can find the code in data/eda.ipynb
*   I did it on local machine and upload it to data/



```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv('./data/eda.csv')
```

## Distributions


```python
plt.figure(figsize=(13,8))
bins=40

plt.subplot(2,2,1)
sns.histplot(data=df, x='x', bins=bins)

plt.subplot(2,2,2)
sns.histplot(data=df, x='y', bins=bins)

plt.subplot(2,2,3)
sns.histplot(data=df, x='width', bins=bins)

plt.subplot(2,2,4)
sns.histplot(data=df, x='height', bins=bins)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fec12bde0d0>




    
![png](darknet_files/darknet_27_1.png)
    


They make sense for number plate images



*   x values are well distributed, which means the cameraman did a good job :D
*   y values are well distributed as well, but, most of the objects are on top of our images.
*   both height and width make sense, because our object is licence plate and they all have almost similiar sizes.







## x vs y | height vs width


```python
plt.figure(figsize=(13,5))

plt.subplot(1,2,1)
sns.scatterplot(data=df, x='x', y='y', alpha=.4)

plt.subplot(1,2,2)
sns.scatterplot(data=df, x='width', y='height', alpha=.4)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fec1259cb10>




    
![png](darknet_files/darknet_30_1.png)
    



1.   As mentioned above, there is a lack in our dataset in buttom-half part of xy plane.
2.   As we can see, the center of our x axis is dense, it's beacuse humans put the object in the center of the camera.



# Darknet Setup


```python
# clone repo
!git clone https://github.com/AlexeyAB/darknet
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
* These files are not in official repo, but you can find them in my repository.


```python
!python generate_train.py
!python generate_test.py
```

## Configs
Wee need to change/create these files: (I configured them for our object, we just need to put them in right place)


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

    /bin/bash: ./darknet: No such file or directory
    

## Restart
I intrupted the training, we can restart training from our last weight.  
(every 100 iterations our weights are saved to backup folder in yolov4-obj_last.weights) (~every 30 minutes)  
(every 1000 iterations our weight are saved to backup folder in yolo-obj_xxxx.weights)


```python
!./darknet detector train ./data/obj.data ./cfg/yolov4-obj.cfg ./backup/yolov4-obj_last.weights -dont_show -map
```

    /bin/bash: ./darknet: No such file or directory
    

# Sanity Check

#### Setup


```python
# set custom cfg to test mode 
%cd cfg
!sed -i 's/batch=64/batch=1/' yolov4-obj.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' yolov4-obj.cfg
%cd ..
```

    /content/drive/MyDrive/darknet/cfg
    /content/drive/My Drive/darknet
    


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

#### Sanity Check on COCO


```python
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg
```

     CUDA-version: 11010 (11020), cuDNN: 7.6.5, CUDNN_HALF=1, GPU count: 1  
     CUDNN_HALF=1 
     OpenCV version: 3.2.0
     0 : compute_capability = 370, cudnn_half = 0, GPU: Tesla K80 
    net.optimized_memory = 0 
    mini_batch = 1, batch = 8, time_steps = 1, train = 0 
       layer   filters  size/strd(dil)      input                output
       0 Create CUDA-stream - 0 
     Create cudnn-handle 0 
    conv     32       3 x 3/ 1    608 x 608 x   3 ->  608 x 608 x  32 0.639 BF
       1 conv     64       3 x 3/ 2    608 x 608 x  32 ->  304 x 304 x  64 3.407 BF
       2 conv     64       1 x 1/ 1    304 x 304 x  64 ->  304 x 304 x  64 0.757 BF
       3 route  1 		                           ->  304 x 304 x  64 
       4 conv     64       1 x 1/ 1    304 x 304 x  64 ->  304 x 304 x  64 0.757 BF
       5 conv     32       1 x 1/ 1    304 x 304 x  64 ->  304 x 304 x  32 0.379 BF
       6 conv     64       3 x 3/ 1    304 x 304 x  32 ->  304 x 304 x  64 3.407 BF
       7 Shortcut Layer: 4,  wt = 0, wn = 0, outputs: 304 x 304 x  64 0.006 BF
       8 conv     64       1 x 1/ 1    304 x 304 x  64 ->  304 x 304 x  64 0.757 BF
       9 route  8 2 	                           ->  304 x 304 x 128 
      10 conv     64       1 x 1/ 1    304 x 304 x 128 ->  304 x 304 x  64 1.514 BF
      11 conv    128       3 x 3/ 2    304 x 304 x  64 ->  152 x 152 x 128 3.407 BF
      12 conv     64       1 x 1/ 1    152 x 152 x 128 ->  152 x 152 x  64 0.379 BF
      13 route  11 		                           ->  152 x 152 x 128 
      14 conv     64       1 x 1/ 1    152 x 152 x 128 ->  152 x 152 x  64 0.379 BF
      15 conv     64       1 x 1/ 1    152 x 152 x  64 ->  152 x 152 x  64 0.189 BF
      16 conv     64       3 x 3/ 1    152 x 152 x  64 ->  152 x 152 x  64 1.703 BF
      17 Shortcut Layer: 14,  wt = 0, wn = 0, outputs: 152 x 152 x  64 0.001 BF
      18 conv     64       1 x 1/ 1    152 x 152 x  64 ->  152 x 152 x  64 0.189 BF
      19 conv     64       3 x 3/ 1    152 x 152 x  64 ->  152 x 152 x  64 1.703 BF
      20 Shortcut Layer: 17,  wt = 0, wn = 0, outputs: 152 x 152 x  64 0.001 BF
      21 conv     64       1 x 1/ 1    152 x 152 x  64 ->  152 x 152 x  64 0.189 BF
      22 route  21 12 	                           ->  152 x 152 x 128 
      23 conv    128       1 x 1/ 1    152 x 152 x 128 ->  152 x 152 x 128 0.757 BF
      24 conv    256       3 x 3/ 2    152 x 152 x 128 ->   76 x  76 x 256 3.407 BF
      25 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
      26 route  24 		                           ->   76 x  76 x 256 
      27 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
      28 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      29 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      30 Shortcut Layer: 27,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      31 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      32 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      33 Shortcut Layer: 30,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      34 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      35 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      36 Shortcut Layer: 33,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      37 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      38 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      39 Shortcut Layer: 36,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      40 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      41 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      42 Shortcut Layer: 39,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      43 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      44 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      45 Shortcut Layer: 42,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      46 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      47 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      48 Shortcut Layer: 45,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      49 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      50 conv    128       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 128 1.703 BF
      51 Shortcut Layer: 48,  wt = 0, wn = 0, outputs:  76 x  76 x 128 0.001 BF
      52 conv    128       1 x 1/ 1     76 x  76 x 128 ->   76 x  76 x 128 0.189 BF
      53 route  52 25 	                           ->   76 x  76 x 256 
      54 conv    256       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 256 0.757 BF
      55 conv    512       3 x 3/ 2     76 x  76 x 256 ->   38 x  38 x 512 3.407 BF
      56 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
      57 route  55 		                           ->   38 x  38 x 512 
      58 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
      59 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      60 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      61 Shortcut Layer: 58,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      62 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      63 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      64 Shortcut Layer: 61,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      65 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      66 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      67 Shortcut Layer: 64,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      68 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      69 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      70 Shortcut Layer: 67,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      71 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      72 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      73 Shortcut Layer: 70,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      74 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      75 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      76 Shortcut Layer: 73,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      77 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      78 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      79 Shortcut Layer: 76,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      80 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      81 conv    256       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 256 1.703 BF
      82 Shortcut Layer: 79,  wt = 0, wn = 0, outputs:  38 x  38 x 256 0.000 BF
      83 conv    256       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 256 0.189 BF
      84 route  83 56 	                           ->   38 x  38 x 512 
      85 conv    512       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 512 0.757 BF
      86 conv   1024       3 x 3/ 2     38 x  38 x 512 ->   19 x  19 x1024 3.407 BF
      87 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
      88 route  86 		                           ->   19 x  19 x1024 
      89 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
      90 conv    512       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.189 BF
      91 conv    512       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x 512 1.703 BF
      92 Shortcut Layer: 89,  wt = 0, wn = 0, outputs:  19 x  19 x 512 0.000 BF
      93 conv    512       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.189 BF
      94 conv    512       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x 512 1.703 BF
      95 Shortcut Layer: 92,  wt = 0, wn = 0, outputs:  19 x  19 x 512 0.000 BF
      96 conv    512       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.189 BF
      97 conv    512       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x 512 1.703 BF
      98 Shortcut Layer: 95,  wt = 0, wn = 0, outputs:  19 x  19 x 512 0.000 BF
      99 conv    512       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.189 BF
     100 conv    512       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x 512 1.703 BF
     101 Shortcut Layer: 98,  wt = 0, wn = 0, outputs:  19 x  19 x 512 0.000 BF
     102 conv    512       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.189 BF
     103 route  102 87 	                           ->   19 x  19 x1024 
     104 conv   1024       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x1024 0.757 BF
     105 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
     106 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
     107 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
     108 max                5x 5/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.005 BF
     109 route  107 		                           ->   19 x  19 x 512 
     110 max                9x 9/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.015 BF
     111 route  107 		                           ->   19 x  19 x 512 
     112 max               13x13/ 1     19 x  19 x 512 ->   19 x  19 x 512 0.031 BF
     113 route  112 110 108 107 	                   ->   19 x  19 x2048 
     114 conv    512       1 x 1/ 1     19 x  19 x2048 ->   19 x  19 x 512 0.757 BF
     115 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
     116 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
     117 conv    256       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 256 0.095 BF
     118 upsample                 2x    19 x  19 x 256 ->   38 x  38 x 256
     119 route  85 		                           ->   38 x  38 x 512 
     120 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     121 route  120 118 	                           ->   38 x  38 x 512 
     122 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     123 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
     124 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     125 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
     126 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     127 conv    128       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 128 0.095 BF
     128 upsample                 2x    38 x  38 x 128 ->   76 x  76 x 128
     129 route  54 		                           ->   76 x  76 x 256 
     130 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
     131 route  130 128 	                           ->   76 x  76 x 256 
     132 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
     133 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
     134 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
     135 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
     136 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
     137 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
     138 conv    255       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 255 0.754 BF
     139 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.20
    nms_kind: greedynms (1), beta = 0.600000 
     140 route  136 		                           ->   76 x  76 x 128 
     141 conv    256       3 x 3/ 2     76 x  76 x 128 ->   38 x  38 x 256 0.852 BF
     142 route  141 126 	                           ->   38 x  38 x 512 
     143 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     144 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
     145 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     146 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
     147 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
     148 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
     149 conv    255       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 255 0.377 BF
     150 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.10
    nms_kind: greedynms (1), beta = 0.600000 
     151 route  147 		                           ->   38 x  38 x 256 
     152 conv    512       3 x 3/ 2     38 x  38 x 256 ->   19 x  19 x 512 0.852 BF
     153 route  152 116 	                           ->   19 x  19 x1024 
     154 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
     155 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
     156 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
     157 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
     158 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
     159 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
     160 conv    255       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 255 0.189 BF
     161 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
    nms_kind: greedynms (1), beta = 0.600000 
    Total BFLOPS 128.459 
    avg_outputs = 1068395 
     Allocate additional workspace_size = 6.65 MB 
    Loading weights from yolov4.weights...
     seen 64, trained: 32032 K-images (500 Kilo-batches_64) 
    Done! Loaded 162 layers from weights-file 
     Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    ./data/person.jpg: Predicted in 171.679000 milli-seconds.
    dog: 99%
    person: 100%
    horse: 98%
    Unable to init server: Could not connect: Connection refused
    
    (predictions:67848): Gtk-[1;33mWARNING[0m **: [34m08:33:32.848[0m: cannot open display: 
    


```python
imShow('./predictions.jpg')
```


    
![png](darknet_files/darknet_58_0.png)
    


#### Run on unseen data
I used the kaggle dataset for this.


```python
!./darknet detector test ./data/obj.data ./cfg/yolov4-obj.cfg ./backup/yolov4-obj_last.weights ../Cars354.png -thresh 0.3
```

    CUDA status Error: file: ./src/dark_cuda.c : () : line: 39 : build time: Sep 20 2021 - 18:31:40 
    
     CUDA Error: no CUDA-capable device is detected
    Darknet error location: ./src/dark_cuda.c, check_error, line #70
    CUDA Error: no CUDA-capable device is detected: Bad file descriptor
    


```python
imShow('./predictions.jpg')
```


    
![png](darknet_files/darknet_61_0.png)
    


**To process a list of images data/train.txt and save results of detection to result.json file use**



```python
!./darknet detector test data/obj.data cfg/yolov4-obj.cfg backup/yolov4-obj_last.weights -ext_output -dont_show -out result.json < data/test.txt
```

     CUDA-version: 11010 (11020), cuDNN: 7.6.5, CUDNN_HALF=1, GPU count: 1  
     CUDNN_HALF=1 
     OpenCV version: 3.2.0
     0 : compute_capability = 370, cudnn_half = 0, GPU: Tesla K80 
    net.optimized_memory = 0 
    mini_batch = 1, batch = 1, time_steps = 1, train = 0 
       layer   filters  size/strd(dil)      input                output
       0 Create CUDA-stream - 0 
     Create cudnn-handle 0 
    conv     32       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  32 0.299 BF
       1 conv     64       3 x 3/ 2    416 x 416 x  32 ->  208 x 208 x  64 1.595 BF
       2 conv     64       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  64 0.354 BF
       3 route  1 		                           ->  208 x 208 x  64 
       4 conv     64       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  64 0.354 BF
       5 conv     32       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  32 0.177 BF
       6 conv     64       3 x 3/ 1    208 x 208 x  32 ->  208 x 208 x  64 1.595 BF
       7 Shortcut Layer: 4,  wt = 0, wn = 0, outputs: 208 x 208 x  64 0.003 BF
       8 conv     64       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  64 0.354 BF
       9 route  8 2 	                           ->  208 x 208 x 128 
      10 conv     64       1 x 1/ 1    208 x 208 x 128 ->  208 x 208 x  64 0.709 BF
      11 conv    128       3 x 3/ 2    208 x 208 x  64 ->  104 x 104 x 128 1.595 BF
      12 conv     64       1 x 1/ 1    104 x 104 x 128 ->  104 x 104 x  64 0.177 BF
      13 route  11 		                           ->  104 x 104 x 128 
      14 conv     64       1 x 1/ 1    104 x 104 x 128 ->  104 x 104 x  64 0.177 BF
      15 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
      16 conv     64       3 x 3/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.797 BF
      17 Shortcut Layer: 14,  wt = 0, wn = 0, outputs: 104 x 104 x  64 0.001 BF
      18 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
      19 conv     64       3 x 3/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.797 BF
      20 Shortcut Layer: 17,  wt = 0, wn = 0, outputs: 104 x 104 x  64 0.001 BF
      21 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
      22 route  21 12 	                           ->  104 x 104 x 128 
      23 conv    128       1 x 1/ 1    104 x 104 x 128 ->  104 x 104 x 128 0.354 BF
      24 conv    256       3 x 3/ 2    104 x 104 x 128 ->   52 x  52 x 256 1.595 BF
      25 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
      26 route  24 		                           ->   52 x  52 x 256 
      27 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
      28 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      29 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      30 Shortcut Layer: 27,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      31 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      32 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      33 Shortcut Layer: 30,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      34 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      35 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      36 Shortcut Layer: 33,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      37 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      38 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      39 Shortcut Layer: 36,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      40 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      41 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      42 Shortcut Layer: 39,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      43 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      44 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      45 Shortcut Layer: 42,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      46 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      47 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      48 Shortcut Layer: 45,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      49 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      50 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      51 Shortcut Layer: 48,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      52 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      53 route  52 25 	                           ->   52 x  52 x 256 
      54 conv    256       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 256 0.354 BF
      55 conv    512       3 x 3/ 2     52 x  52 x 256 ->   26 x  26 x 512 1.595 BF
      56 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
      57 route  55 		                           ->   26 x  26 x 512 
      58 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
      59 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      60 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      61 Shortcut Layer: 58,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      62 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      63 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      64 Shortcut Layer: 61,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      65 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      66 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      67 Shortcut Layer: 64,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      68 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      69 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      70 Shortcut Layer: 67,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      71 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      72 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      73 Shortcut Layer: 70,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      74 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      75 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      76 Shortcut Layer: 73,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      77 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      78 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      79 Shortcut Layer: 76,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      80 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      81 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      82 Shortcut Layer: 79,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      83 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      84 route  83 56 	                           ->   26 x  26 x 512 
      85 conv    512       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 512 0.354 BF
      86 conv   1024       3 x 3/ 2     26 x  26 x 512 ->   13 x  13 x1024 1.595 BF
      87 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
      88 route  86 		                           ->   13 x  13 x1024 
      89 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
      90 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
      91 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      92 Shortcut Layer: 89,  wt = 0, wn = 0, outputs:  13 x  13 x 512 0.000 BF
      93 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
      94 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      95 Shortcut Layer: 92,  wt = 0, wn = 0, outputs:  13 x  13 x 512 0.000 BF
      96 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
      97 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      98 Shortcut Layer: 95,  wt = 0, wn = 0, outputs:  13 x  13 x 512 0.000 BF
      99 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
     100 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
     101 Shortcut Layer: 98,  wt = 0, wn = 0, outputs:  13 x  13 x 512 0.000 BF
     102 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
     103 route  102 87 	                           ->   13 x  13 x1024 
     104 conv   1024       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x1024 0.354 BF
     105 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     106 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     107 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     108 max                5x 5/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.002 BF
     109 route  107 		                           ->   13 x  13 x 512 
     110 max                9x 9/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.007 BF
     111 route  107 		                           ->   13 x  13 x 512 
     112 max               13x13/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.015 BF
     113 route  112 110 108 107 	                   ->   13 x  13 x2048 
     114 conv    512       1 x 1/ 1     13 x  13 x2048 ->   13 x  13 x 512 0.354 BF
     115 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     116 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     117 conv    256       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 256 0.044 BF
     118 upsample                 2x    13 x  13 x 256 ->   26 x  26 x 256
     119 route  85 		                           ->   26 x  26 x 512 
     120 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     121 route  120 118 	                           ->   26 x  26 x 512 
     122 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     123 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     124 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     125 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     126 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     127 conv    128       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 128 0.044 BF
     128 upsample                 2x    26 x  26 x 128 ->   52 x  52 x 128
     129 route  54 		                           ->   52 x  52 x 256 
     130 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
     131 route  130 128 	                           ->   52 x  52 x 256 
     132 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
     133 conv    256       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 256 1.595 BF
     134 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
     135 conv    256       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 256 1.595 BF
     136 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
     137 conv    256       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 256 1.595 BF
     138 conv     18       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x  18 0.025 BF
     139 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.20
    nms_kind: greedynms (1), beta = 0.600000 
     140 route  136 		                           ->   52 x  52 x 128 
     141 conv    256       3 x 3/ 2     52 x  52 x 128 ->   26 x  26 x 256 0.399 BF
     142 route  141 126 	                           ->   26 x  26 x 512 
     143 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     144 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     145 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     146 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     147 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     148 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     149 conv     18       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x  18 0.012 BF
     150 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.10
    nms_kind: greedynms (1), beta = 0.600000 
     151 route  147 		                           ->   26 x  26 x 256 
     152 conv    512       3 x 3/ 2     26 x  26 x 256 ->   13 x  13 x 512 0.399 BF
     153 route  152 116 	                           ->   13 x  13 x1024 
     154 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     155 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     156 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     157 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     158 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     159 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     160 conv     18       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x  18 0.006 BF
     161 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
    nms_kind: greedynms (1), beta = 0.600000 
    Total BFLOPS 59.563 
    avg_outputs = 489778 
     Allocate additional workspace_size = 12.46 MB 
    Loading weights from backup/yolov4-obj_last.weights...
     seen 64, trained: 38 K-images (0 Kilo-batches_64) 
    Done! Loaded 162 layers from weights-file 
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b15d6c0bdf90226d.jpg: Predicted in 100.508000 milli-seconds.
    license_plate: 70%	(left_x:  868   top_y:  508   width:  101   height:   82)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a6e3b0b73220cd32.jpg: Predicted in 96.745000 milli-seconds.
    license_plate: 31%	(left_x:  427   top_y:  425   width:  185   height:   55)
    license_plate: 77%	(left_x:  435   top_y:  444   width:  180   height:   27)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d7d49b1a3706f3eb.jpg: Predicted in 96.334000 milli-seconds.
    license_plate: 35%	(left_x:  178   top_y:  419   width:   95   height:   41)
    license_plate: 30%	(left_x:  178   top_y:  435   width:  140   height:   34)
    license_plate: 30%	(left_x:  423   top_y:  428   width:   82   height:   30)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/99732f172ac80128.jpg: Predicted in 90.732000 milli-seconds.
    license_plate: 61%	(left_x:  675   top_y:  264   width:  103   height:   37)
    license_plate: 29%	(left_x:  698   top_y:  316   width:  130   height:   71)
    license_plate: 55%	(left_x:  745   top_y:  329   width:   58   height:   48)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f5128e7a123b4fa8.jpg: Predicted in 90.922000 milli-seconds.
    license_plate: 59%	(left_x:  377   top_y:  381   width:  125   height:   34)
    license_plate: 35%	(left_x:  381   top_y:  384   width:   87   height:   23)
    license_plate: 75%	(left_x:  561   top_y:  374   width:  171   height:   43)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/620bd952592bc0e9.jpg: Predicted in 90.396000 milli-seconds.
    license_plate: 35%	(left_x:  945   top_y:  307   width:   61   height:   14)
    license_plate: 30%	(left_x:  955   top_y:  294   width:   41   height:   24)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/bacc80fafc0566c3.jpg: Predicted in 90.677000 milli-seconds.
    license_plate: 66%	(left_x:  779   top_y:  375   width:   37   height:   34)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/99c9c657582f958f.jpg: Predicted in 90.837000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b193070a9c45b5ab.jpg: Predicted in 90.501000 milli-seconds.
    license_plate: 41%	(left_x:  201   top_y:  449   width:   96   height:   33)
    license_plate: 82%	(left_x:  215   top_y:  434   width:   82   height:   63)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/35875c388efcddf0.jpg: Predicted in 90.591000 milli-seconds.
    license_plate: 57%	(left_x:  467   top_y:  324   width:  256   height:  232)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2d40d6e377c82ec9.jpg: Predicted in 90.941000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4ac941b9393d00f9.jpg: Predicted in 90.616000 milli-seconds.
    license_plate: 35%	(left_x: 1000   top_y:  417   width:   23   height:   28)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e2427db217e5c3db.jpg: Predicted in 90.696000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e4d600a63db8bf3a.jpg: Predicted in 90.429000 milli-seconds.
    license_plate: 43%	(left_x:  879   top_y:  440   width:  121   height:  164)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/26192da5ce2afa62.jpg: Predicted in 90.992000 milli-seconds.
    license_plate: 43%	(left_x:  331   top_y:   73   width:   76   height:   24)
    license_plate: 42%	(left_x:  348   top_y:   70   width:   69   height:   34)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/133921bd26fff729.jpg: Predicted in 90.875000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/747b0f12f54703ee.jpg: Predicted in 90.584000 milli-seconds.
    license_plate: 36%	(left_x:  367   top_y:  542   width:  322   height:   51)
    license_plate: 61%	(left_x:  443   top_y:  536   width:  155   height:   68)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/62bb93bbd270dd9a.jpg: Predicted in 90.833000 milli-seconds.
    license_plate: 49%	(left_x:  577   top_y:  403   width:  234   height:   59)
    license_plate: 64%	(left_x:  660   top_y:  412   width:   87   height:   33)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2b97f5bf137ee8d1.jpg: Predicted in 91.025000 milli-seconds.
    license_plate: 41%	(left_x:   88   top_y:  299   width:   40   height:   20)
    license_plate: 45%	(left_x:  143   top_y:  434   width:   85   height:   34)
    license_plate: 34%	(left_x:  945   top_y:  369   width:   63   height:   37)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2619ec27a314e69c.jpg: Predicted in 89.486000 milli-seconds.
    license_plate: 42%	(left_x:   34   top_y:  470   width:  116   height:   86)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e9cd346a4a84d594.jpg: Predicted in 88.808000 milli-seconds.
    license_plate: 91%	(left_x:  280   top_y:  480   width:  191   height:   94)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/9034927e7438bdd6.jpg: Predicted in 88.689000 milli-seconds.
    license_plate: 40%	(left_x:  160   top_y:  774   width:  112   height:   62)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4ff8bebd6d6a0361.jpg: Predicted in 88.701000 milli-seconds.
    license_plate: 67%	(left_x:  398   top_y:  360   width:  259   height:   97)
    license_plate: 41%	(left_x:  764   top_y:  428   width:  110   height:   43)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/de2b5afb7eda96cf.jpg: Predicted in 88.748000 milli-seconds.
    license_plate: 80%	(left_x:  763   top_y:  412   width:  111   height:   39)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/53925df03b471f5d.jpg: Predicted in 88.974000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/11b155ab5b3331cf.jpg: Predicted in 89.009000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/28d1fb27a6e42ee7.jpg: Predicted in 88.973000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/90596bf3313e72e3.jpg: Predicted in 89.038000 milli-seconds.
    license_plate: 52%	(left_x:  105   top_y:  217   width:  135   height:   25)
    license_plate: 79%	(left_x:  140   top_y:  208   width:   65   height:   34)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/c650ff8d3e8e75b3.jpg: Predicted in 89.136000 milli-seconds.
    license_plate: 34%	(left_x:   50   top_y:  467   width:  190   height:   58)
    license_plate: 53%	(left_x:  100   top_y:  457   width:   94   height:   80)
    license_plate: 52%	(left_x:  112   top_y:  447   width:   70   height:   69)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/53386dae3cf13cc3.jpg: Predicted in 88.765000 milli-seconds.
    license_plate: 71%	(left_x:  305   top_y:  501   width:  288   height:   70)
    license_plate: 29%	(left_x:  382   top_y:  449   width:  147   height:  210)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/11f80fda2c38011c.jpg: Predicted in 89.086000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/fe6139d150a3e2a8.jpg: Predicted in 88.788000 milli-seconds.
    license_plate: 26%	(left_x:  358   top_y:  722   width:  274   height:   75)
    license_plate: 54%	(left_x:  499   top_y:  206   width:  244   height:   46)
    license_plate: 25%	(left_x:  557   top_y:  219   width:  161   height:   26)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d6c5271e96ec1a61.jpg: Predicted in 88.948000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/87793700161a7a6b.jpg: Predicted in 88.903000 milli-seconds.
    license_plate: 48%	(left_x:  387   top_y:  449   width:  375   height:   83)
    license_plate: 71%	(left_x:  472   top_y:  458   width:  209   height:   60)
    license_plate: 69%	(left_x:  543   top_y:  443   width:   80   height:   92)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2f3887024d298547.jpg: Predicted in 88.899000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b8a3f2ea385e45b3.jpg: Predicted in 89.006000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/66be6b583048fa94.jpg: Predicted in 86.254000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/1545c73bdecb3e2f.jpg: Predicted in 82.881000 milli-seconds.
    license_plate: 46%	(left_x:   98   top_y:  340   width:  100   height:   31)
    license_plate: 59%	(left_x:  112   top_y:  325   width:   63   height:   52)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/43339d9cbca470e4.jpg: Predicted in 81.816000 milli-seconds.
    license_plate: 69%	(left_x:  596   top_y:  276   width:  171   height:   56)
    license_plate: 56%	(left_x:  623   top_y:  251   width:  121   height:  110)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/c7b1e6f7c38fa1a0.jpg: Predicted in 74.126000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/8cb5553e75d3d311.jpg: Predicted in 74.200000 milli-seconds.
    license_plate: 87%	(left_x:  100   top_y:  340   width:   77   height:   73)
    license_plate: 58%	(left_x:  109   top_y:  325   width:   57   height:  124)
    license_plate: 44%	(left_x:  269   top_y:  150   width:   69   height:   39)
    license_plate: 60%	(left_x:  278   top_y:  159   width:   50   height:   17)
    license_plate: 80%	(left_x:  836   top_y:  143   width:   79   height:   26)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/88df55ff4e13b90d.jpg: Predicted in 74.230000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/724307be418b2ed2.jpg: Predicted in 73.945000 milli-seconds.
    license_plate: 69%	(left_x:  186   top_y:  429   width:  142   height:   36)
    license_plate: 64%	(left_x:  210   top_y:  414   width:  111   height:   82)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/8c63cf76166c3bd7.jpg: Predicted in 74.102000 milli-seconds.
    license_plate: 28%	(left_x:  715   top_y:  462   width:  182   height:   47)
    license_plate: 27%	(left_x:  753   top_y:  449   width:   87   height:   42)
    license_plate: 45%	(left_x:  761   top_y:  447   width:   68   height:   72)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/fb55b73f241bf50a.jpg: Predicted in 73.079000 milli-seconds.
    license_plate: 39%	(left_x:  421   top_y:  368   width:  160   height:  151)
    license_plate: 44%	(left_x:  685   top_y:  405   width:  189   height:   73)
    license_plate: 32%	(left_x:  720   top_y:  378   width:  114   height:  120)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4b3aba5dfb7a0492.jpg: Predicted in 71.903000 milli-seconds.
    license_plate: 31%	(left_x:  123   top_y:  351   width:  183   height:  149)
    license_plate: 37%	(left_x:  156   top_y:  391   width:  140   height:   82)
    license_plate: 87%	(left_x:  196   top_y:  396   width:   56   height:   71)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/185d5dfa193c4ced.jpg: Predicted in 71.771000 milli-seconds.
    license_plate: 96%	(left_x:  408   top_y:  307   width:  170   height:   80)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/057569768fd6303e.jpg: Predicted in 71.737000 milli-seconds.
    license_plate: 53%	(left_x:   16   top_y:  245   width:   66   height:   23)
    license_plate: 26%	(left_x:   20   top_y:  231   width:   61   height:   30)
    license_plate: 75%	(left_x:  863   top_y:  444   width:  105   height:   95)
    license_plate: 46%	(left_x:  899   top_y:  249   width:  113   height:   50)
    license_plate: 57%	(left_x:  925   top_y:  254   width:   58   height:   35)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f3814f4a6121838d.jpg: Predicted in 71.774000 milli-seconds.
    license_plate: 32%	(left_x:  651   top_y:  221   width:  190   height:   83)
    license_plate: 51%	(left_x:  675   top_y:  238   width:  133   height:   42)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/8256e277c7f47797.jpg: Predicted in 72.087000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e9561d7c16db565f.jpg: Predicted in 71.729000 milli-seconds.
    license_plate: 71%	(left_x:  201   top_y:  345   width:  139   height:   45)
    license_plate: 62%	(left_x:  217   top_y:  320   width:   99   height:   90)
    license_plate: 33%	(left_x:  221   top_y:  357   width:   91   height:   40)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/90abd4265adf5a00.jpg: Predicted in 71.779000 milli-seconds.
    license_plate: 31%	(left_x:   65   top_y:  443   width:   46   height:   14)
    license_plate: 62%	(left_x:   71   top_y:  429   width:   35   height:   23)
    license_plate: 34%	(left_x:  410   top_y:  458   width:   60   height:   17)
    license_plate: 63%	(left_x:  418   top_y:  433   width:   44   height:   12)
    license_plate: 53%	(left_x:  428   top_y:  454   width:   29   height:   21)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/6c144e47161867c2.jpg: Predicted in 72.020000 milli-seconds.
    license_plate: 43%	(left_x:  319   top_y:  440   width:  176   height:   47)
    license_plate: 25%	(left_x:  337   top_y:  443   width:  101   height:   36)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/8cdcb833ef1ef049.jpg: Predicted in 71.661000 milli-seconds.
    license_plate: 53%	(left_x:  184   top_y:  355   width:  242   height:   46)
    license_plate: 36%	(left_x:  247   top_y:  353   width:  112   height:   42)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/593e594137f374ab.jpg: Predicted in 71.896000 milli-seconds.
    license_plate: 46%	(left_x:  358   top_y:  398   width:  645   height:  102)
    license_plate: 31%	(left_x:  523   top_y:  377   width:  338   height:  117)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/bf0aac0878b0a3d2.jpg: Predicted in 73.061000 milli-seconds.
    license_plate: 70%	(left_x:  632   top_y:  432   width:  120   height:   74)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/feaf383fc5de383f.jpg: Predicted in 71.679000 milli-seconds.
    license_plate: 64%	(left_x:  383   top_y:  688   width:  226   height:   64)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/415d64bf8cfe82f2.jpg: Predicted in 71.950000 milli-seconds.
    license_plate: 57%	(left_x:  367   top_y:  333   width:  268   height:   56)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f9170e8c13a99991.jpg: Predicted in 71.903000 milli-seconds.
    license_plate: 62%	(left_x:  397   top_y:   77   width:  224   height:  136)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/008637722500f239.jpg: Predicted in 71.801000 milli-seconds.
    license_plate: 64%	(left_x:  252   top_y:  396   width:   95   height:   46)
    license_plate: 61%	(left_x:  804   top_y:  257   width:   72   height:   24)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5641e6ad7d2600f2.jpg: Predicted in 72.073000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5a96f5e1f9be8d42.jpg: Predicted in 71.861000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4572990fd64bb6be.jpg: Predicted in 71.886000 milli-seconds.
    license_plate: 43%	(left_x:    5   top_y:  234   width:   81   height:   49)
    license_plate: 55%	(left_x:   10   top_y:  237   width:   45   height:   42)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4ee3f0e9a2fb20a9.jpg: Predicted in 72.200000 milli-seconds.
    license_plate: 79%	(left_x:  300   top_y:  286   width:  209   height:  139)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/53ad98b12752ad16.jpg: Predicted in 71.597000 milli-seconds.
    license_plate: 31%	(left_x:  114   top_y:  170   width:   59   height:   30)
    license_plate: 50%	(left_x:  121   top_y:  174   width:   48   height:   15)
    license_plate: 56%	(left_x:  239   top_y:  544   width:  113   height:   64)
    license_plate: 78%	(left_x:  354   top_y:  336   width:  302   height:  116)
    license_plate: 86%	(left_x:  446   top_y:  349   width:  114   height:   84)
    license_plate: 31%	(left_x:  448   top_y:  320   width:  133   height:  147)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f3d472955c13cae0.jpg: Predicted in 71.854000 milli-seconds.
    license_plate: 43%	(left_x:  393   top_y:  581   width:  210   height:   52)
    license_plate: 41%	(left_x:  454   top_y:  577   width:   91   height:   44)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f5d1729aa333b284.jpg: Predicted in 71.874000 milli-seconds.
    license_plate: 31%	(left_x:   74   top_y:  237   width:   73   height:   39)
    license_plate: 47%	(left_x:   86   top_y:  243   width:   48   height:   27)
    license_plate: 80%	(left_x:  702   top_y:  505   width:  130   height:   88)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d565d93637d4e76d.jpg: Predicted in 72.051000 milli-seconds.
    license_plate: 30%	(left_x:  315   top_y:  433   width:   98   height:   41)
    license_plate: 27%	(left_x:  337   top_y:  436   width:   55   height:   44)
    license_plate: 28%	(left_x:  398   top_y:  481   width:  371   height:   78)
    license_plate: 41%	(left_x:  508   top_y:  486   width:  181   height:   57)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f53243c2bb551b8a.jpg: Predicted in 71.947000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/3ecd5af7c4f11963.jpg: Predicted in 71.748000 milli-seconds.
    license_plate: 51%	(left_x:  202   top_y:  257   width:  499   height:  173)
    license_plate: 52%	(left_x:  368   top_y:  245   width:  180   height:  192)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/182268e1f8c6525f.jpg: Predicted in 71.902000 milli-seconds.
    license_plate: 57%	(left_x:  481   top_y:  400   width:  178   height:   99)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f38265abb22a00e4.jpg: Predicted in 71.601000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b101900b26128253.jpg: Predicted in 72.045000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/621837d55c229864.jpg: Predicted in 71.765000 milli-seconds.
    license_plate: 59%	(left_x:  293   top_y:  565   width:  107   height:   33)
    license_plate: 35%	(left_x:  319   top_y:  410   width:  159   height:   88)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e79f6777f2fc08b2.jpg: Predicted in 71.789000 milli-seconds.
    license_plate: 82%	(left_x:  391   top_y:  447   width:  271   height:   83)
    license_plate: 52%	(left_x:  481   top_y:  452   width:   96   height:   68)
    license_plate: 78%	(left_x:  487   top_y:  419   width:  159   height:  140)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/fc0a607ba5f38505.jpg: Predicted in 71.099000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4cb48c8bf41b70a4.jpg: Predicted in 68.944000 milli-seconds.
    license_plate: 67%	(left_x:  372   top_y:  397   width:  305   height:  122)
    license_plate: 69%	(left_x:  411   top_y:  431   width:  236   height:   60)
    license_plate: 95%	(left_x:  438   top_y:  406   width:  151   height:  113)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/88fc866f92860d70.jpg: Predicted in 68.681000 milli-seconds.
    license_plate: 41%	(left_x:  411   top_y:  673   width:  393   height:  154)
    license_plate: 73%	(left_x:  532   top_y:  706   width:  175   height:  102)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/ae6724423f0d111a.jpg: Predicted in 68.707000 milli-seconds.
    license_plate: 77%	(left_x:  611   top_y:  622   width:  222   height:   71)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f5c2a80a21ddb78c.jpg: Predicted in 68.798000 milli-seconds.
    license_plate: 84%	(left_x:  178   top_y:  226   width:  223   height:   41)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b57fe3aa2ff9577e.jpg: Predicted in 68.569000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/80d21c58a71a0751.jpg: Predicted in 68.838000 milli-seconds.
    license_plate: 39%	(left_x:  252   top_y:  308   width:  192   height:   62)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/548a8e997a66a0c4.jpg: Predicted in 68.954000 milli-seconds.
    license_plate: 42%	(left_x:  399   top_y:  421   width:  157   height:   75)
    license_plate: 31%	(left_x:  446   top_y:  443   width:   67   height:   35)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/02a6ef3d9bd68e91.jpg: Predicted in 68.780000 milli-seconds.
    license_plate: 50%	(left_x:  142   top_y:  311   width:   83   height:   35)
    license_plate: 52%	(left_x:  155   top_y:  318   width:   55   height:   24)
    license_plate: 54%	(left_x:  937   top_y:  117   width:   77   height:   11)
    license_plate: 43%	(left_x:  955   top_y:  115   width:   40   height:   21)
    license_plate: 58%	(left_x:  963   top_y:  113   width:   58   height:   22)
    license_plate: 39%	(left_x:  972   top_y:  109   width:   39   height:   35)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/c274e026ac907eb9.jpg: Predicted in 68.598000 milli-seconds.
    license_plate: 47%	(left_x:  273   top_y:  333   width:  300   height:  133)
    license_plate: 49%	(left_x:  379   top_y:  367   width:  130   height:   67)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/6e819dc674a74078.jpg: Predicted in 68.831000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0cacb08195a3e2d7.jpg: Predicted in 68.634000 milli-seconds.
    license_plate: 91%	(left_x:  290   top_y:  391   width:  161   height:   79)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/894cacdb385fdb6b.jpg: Predicted in 68.648000 milli-seconds.
    license_plate: 39%	(left_x:  484   top_y:  541   width:  112   height:   50)
    license_plate: 78%	(left_x:  490   top_y:  522   width:  102   height:   58)
    license_plate: 28%	(left_x:  503   top_y:  538   width:   80   height:   29)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/485e5f37dd13ffab.jpg: Predicted in 68.619000 milli-seconds.
    license_plate: 32%	(left_x:  279   top_y:  348   width:  111   height:   46)
    license_plate: 36%	(left_x:  296   top_y:  363   width:  100   height:   36)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/27798906120ea394.jpg: Predicted in 68.747000 milli-seconds.
    license_plate: 69%	(left_x:  153   top_y:  407   width:  197   height:   65)
    license_plate: 39%	(left_x:  189   top_y:  421   width:  129   height:   57)
    license_plate: 32%	(left_x:  884   top_y:  173   width:   96   height:   33)
    license_plate: 38%	(left_x:  911   top_y:  181   width:   46   height:   19)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0673b967f8c68eec.jpg: Predicted in 68.680000 milli-seconds.
    license_plate: 64%	(left_x:  241   top_y:  534   width:  205   height:   96)
    license_plate: 27%	(left_x:  295   top_y:  489   width:  143   height:  200)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/9a795bbf4e94d630.jpg: Predicted in 68.899000 milli-seconds.
    license_plate: 29%	(left_x:  291   top_y:  464   width:  182   height:   59)
    license_plate: 32%	(left_x:  329   top_y:  483   width:  151   height:   44)
    license_plate: 36%	(left_x:  360   top_y:  450   width:  110   height:   89)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/081f5a6bc61b9c48.jpg: Predicted in 68.683000 milli-seconds.
    license_plate: 85%	(left_x:  572   top_y:  342   width:  162   height:   76)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/908a3ac555a1a509.jpg: Predicted in 69.343000 milli-seconds.
    license_plate: 30%	(left_x:  587   top_y:  480   width:  108   height:   25)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d9fa2abf3719a4bd.jpg: Predicted in 68.713000 milli-seconds.
    license_plate: 82%	(left_x:   32   top_y:  375   width:  198   height:   95)
    license_plate: 52%	(left_x:  410   top_y:  407   width:  236   height:   83)
    license_plate: 36%	(left_x:  449   top_y:  414   width:  132   height:   57)
    license_plate: 46%	(left_x:  459   top_y:  428   width:  148   height:   52)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/76e14d05ee8acb6d.jpg: Predicted in 68.705000 milli-seconds.
    license_plate: 70%	(left_x:   80   top_y:  306   width:  192   height:   64)
    license_plate: 81%	(left_x:  126   top_y:  282   width:  114   height:  128)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/fdd963edc28bf163.jpg: Predicted in 68.754000 milli-seconds.
    license_plate: 26%	(left_x:  233   top_y:  356   width:  198   height:   72)
    license_plate: 28%	(left_x:  293   top_y:  321   width:   86   height:  107)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e742e284c7c9da96.jpg: Predicted in 68.627000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4fa24c1abe969cb7.jpg: Predicted in 68.767000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4e593c88022ff6b1.jpg: Predicted in 68.860000 milli-seconds.
    license_plate: 51%	(left_x:  550   top_y:  418   width:  201   height:  127)
    license_plate: 54%	(left_x:  573   top_y:  448   width:  159   height:   67)
    license_plate: 26%	(left_x:  742   top_y:  212   width:  109   height:   71)
    license_plate: 77%	(left_x:  753   top_y:  234   width:   84   height:   45)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4793138df3c05610.jpg: Predicted in 68.775000 milli-seconds.
    license_plate: 78%	(left_x:  469   top_y:  394   width:  122   height:   28)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/1eb2ebab9cd7adf6.jpg: Predicted in 68.701000 milli-seconds.
    license_plate: 26%	(left_x:  419   top_y:  394   width:   87   height:   24)
    license_plate: 39%	(left_x:  437   top_y:  396   width:   55   height:   17)
    license_plate: 33%	(left_x:  449   top_y:  399   width:   55   height:   16)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d027c6e32db60e3c.jpg: Predicted in 68.468000 milli-seconds.
    license_plate: 38%	(left_x:  272   top_y:  381   width:  456   height:  116)
    license_plate: 94%	(left_x:  408   top_y:  409   width:  198   height:   73)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/ed7d99098d4af8e6.jpg: Predicted in 69.031000 milli-seconds.
    license_plate: 28%	(left_x:  100   top_y:  389   width:  207   height:   39)
    license_plate: 57%	(left_x:  129   top_y:  392   width:  120   height:   30)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2a3e44107826e876.jpg: Predicted in 68.877000 milli-seconds.
    license_plate: 64%	(left_x:  586   top_y:  232   width:  150   height:   79)
    license_plate: 40%	(left_x:  603   top_y:  238   width:  115   height:   50)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/934645379fe657e1.jpg: Predicted in 69.047000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2bd26c63ebf598b7.jpg: Predicted in 68.902000 milli-seconds.
    license_plate: 80%	(left_x:  641   top_y:  345   width:  177   height:   82)
    license_plate: 60%	(left_x:  661   top_y:  383   width:  137   height:   31)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/3e2306f2cf4b2c67.jpg: Predicted in 68.262000 milli-seconds.
    license_plate: 40%	(left_x:  560   top_y:  516   width:  150   height:   87)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5a3f94aa35766f31.jpg: Predicted in 67.256000 milli-seconds.
    license_plate: 34%	(left_x:  682   top_y:  324   width:  114   height:   35)
    license_plate: 60%	(left_x:  691   top_y:  306   width:   87   height:   65)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f1131b93a33cefb9.jpg: Predicted in 67.166000 milli-seconds.
    license_plate: 64%	(left_x:  383   top_y:  489   width:  192   height:   82)
    license_plate: 27%	(left_x:  403   top_y:  476   width:   98   height:  137)
    license_plate: 29%	(left_x:  443   top_y:  364   width:   78   height:  362)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2adddd0b09bb0f17.jpg: Predicted in 67.407000 milli-seconds.
    license_plate: 35%	(left_x:  385   top_y:  371   width:  265   height:   65)
    license_plate: 70%	(left_x:  432   top_y:  346   width:  180   height:  129)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/ad10c7d29f111692.jpg: Predicted in 67.127000 milli-seconds.
    license_plate: 72%	(left_x:  759   top_y:  574   width:  112   height:   66)
    license_plate: 67%	(left_x:  760   top_y:  592   width:  152   height:   39)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/6504632e0dc25997.jpg: Predicted in 67.353000 milli-seconds.
    license_plate: 55%	(left_x:  153   top_y:  351   width:   66   height:   23)
    license_plate: 46%	(left_x:  707   top_y:  575   width:  176   height:   48)
    license_plate: 41%	(left_x:  749   top_y:  545   width:   90   height:  108)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/1f1d5bc31e444cc1.jpg: Predicted in 67.279000 milli-seconds.
    license_plate: 53%	(left_x:  303   top_y:  363   width:  233   height:   48)
    license_plate: 69%	(left_x:  350   top_y:  351   width:  138   height:   73)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/7a1f9520c7ecd5b1.jpg: Predicted in 67.240000 milli-seconds.
    license_plate: 46%	(left_x:   86   top_y:  296   width:  164   height:   42)
    license_plate: 26%	(left_x:  108   top_y:  305   width:  120   height:   22)
    license_plate: 39%	(left_x:  784   top_y:  589   width:  144   height:   44)
    license_plate: 66%	(left_x:  809   top_y:  583   width:  100   height:   32)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/1db3793d7c84faa1.jpg: Predicted in 67.428000 milli-seconds.
    license_plate: 47%	(left_x:  160   top_y:  417   width:   55   height:   35)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2956bcbedfc167b6.jpg: Predicted in 67.177000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5021041c726b6ee4.jpg: Predicted in 67.219000 milli-seconds.
    license_plate: 49%	(left_x:  488   top_y:  367   width:  166   height:   82)
    license_plate: 58%	(left_x:  521   top_y:  380   width:  101   height:   59)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2e95d7a799e23e11.jpg: Predicted in 67.558000 milli-seconds.
    license_plate: 90%	(left_x:  734   top_y:  269   width:  145   height:   44)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/3ed000628c6a0587.jpg: Predicted in 67.391000 milli-seconds.
    license_plate: 74%	(left_x:  405   top_y:  363   width:  159   height:   63)
    license_plate: 69%	(left_x:  438   top_y:  376   width:   87   height:   44)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a79296fb4a48245e.jpg: Predicted in 67.108000 milli-seconds.
    license_plate: 43%	(left_x:  181   top_y:  385   width:  210   height:   53)
    license_plate: 74%	(left_x:  235   top_y:  407   width:  135   height:   30)
    license_plate: 57%	(left_x:  249   top_y:  394   width:   92   height:   58)
    license_plate: 30%	(left_x:  638   top_y:  296   width:   39   height:   25)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/15a51e29f5ceffd8.jpg: Predicted in 67.302000 milli-seconds.
    license_plate: 52%	(left_x:  434   top_y:  404   width:  131   height:   60)
    license_plate: 59%	(left_x:  438   top_y:  377   width:  156   height:  114)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/7cbc7ee444622a66.jpg: Predicted in 67.242000 milli-seconds.
    license_plate: 26%	(left_x:  467   top_y:  616   width:  207   height:   81)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a823532e163f5722.jpg: Predicted in 67.367000 milli-seconds.
    license_plate: 45%	(left_x:  270   top_y:  563   width:  192   height:  104)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b25d7d2cb7abf86b.jpg: Predicted in 67.408000 milli-seconds.
    license_plate: 46%	(left_x:  367   top_y:  367   width:  306   height:  239)
    license_plate: 63%	(left_x:  410   top_y:  437   width:  217   height:   77)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/c7df5232291486a3.jpg: Predicted in 67.674000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/affd16c51644a893.jpg: Predicted in 67.384000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/6aff04e9c32aec0f.jpg: Predicted in 67.294000 milli-seconds.
    license_plate: 32%	(left_x:  -11   top_y:  693   width:  177   height:  111)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/23c2c79a9febbf3e.jpg: Predicted in 67.553000 milli-seconds.
    license_plate: 63%	(left_x:  632   top_y:  372   width:  108   height:   63)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f83d18919353c483.jpg: Predicted in 67.399000 milli-seconds.
    license_plate: 26%	(left_x:  270   top_y:  452   width:  115   height:   32)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/7fc0fb6035bc06a6.jpg: Predicted in 67.333000 milli-seconds.
    license_plate: 55%	(left_x:  404   top_y:  524   width:  152   height:   29)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/508f0a75156e5fd4.jpg: Predicted in 67.137000 milli-seconds.
    license_plate: 26%	(left_x:  878   top_y:  449   width:  105   height:   63)
    license_plate: 39%	(left_x:  911   top_y:  469   width:   47   height:   20)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/06b024413ad385a7.jpg: Predicted in 67.285000 milli-seconds.
    license_plate: 44%	(left_x:  816   top_y:  448   width:  134   height:   88)
    license_plate: 63%	(left_x:  834   top_y:  484   width:   86   height:   35)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/017527da8bfeb97d.jpg: Predicted in 67.116000 milli-seconds.
    license_plate: 61%	(left_x:  491   top_y:  291   width:  322   height:  106)
    license_plate: 82%	(left_x:  566   top_y:  307   width:  211   height:   74)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a774a6f81fea258b.jpg: Predicted in 67.308000 milli-seconds.
    license_plate: 55%	(left_x:  804   top_y:  361   width:  110   height:   79)
    license_plate: 72%	(left_x:  814   top_y:  372   width:   90   height:   42)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/7f56adf4b9306ac9.jpg: Predicted in 67.332000 milli-seconds.
    license_plate: 83%	(left_x:  175   top_y:  309   width:  240   height:   49)
    license_plate: 28%	(left_x:  283   top_y:  294   width:   32   height:   68)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/8cca720c410ee7f8.jpg: Predicted in 67.427000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4148b2126f0986a4.jpg: Predicted in 67.375000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/ec8a36f7874c0c34.jpg: Predicted in 67.438000 milli-seconds.
    license_plate: 71%	(left_x:  194   top_y:  392   width:  111   height:   23)
    license_plate: 71%	(left_x:  199   top_y:  374   width:  112   height:   49)
    license_plate: 26%	(left_x:  205   top_y:  343   width:   82   height:   32)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/69450fa183d57a7b.jpg: Predicted in 67.273000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d44c0b2252bf8a92.jpg: Predicted in 67.376000 milli-seconds.
    license_plate: 52%	(left_x:  345   top_y:  389   width:  272   height:   44)
    license_plate: 31%	(left_x:  377   top_y:  374   width:  211   height:   88)
    license_plate: 26%	(left_x:  417   top_y:  382   width:  126   height:   53)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/aca8821ff0368720.jpg: Predicted in 67.323000 milli-seconds.
    license_plate: 46%	(left_x:  314   top_y:  169   width:   99   height:   46)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0787b0fa95f545a5.jpg: Predicted in 67.378000 milli-seconds.
    license_plate: 55%	(left_x:  424   top_y:  504   width:  198   height:   70)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/19ff847d0cd1c5ec.jpg: Predicted in 67.166000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/aefe747315dd79fc.jpg: Predicted in 67.230000 milli-seconds.
    license_plate: 56%	(left_x:  213   top_y:  509   width:  169   height:  100)
    license_plate: 33%	(left_x:  269   top_y:  492   width:  105   height:  148)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/67945cf7a6beccdf.jpg: Predicted in 67.297000 milli-seconds.
    license_plate: 71%	(left_x:  400   top_y:  436   width:  254   height:   41)
    license_plate: 28%	(left_x:  495   top_y:  429   width:   73   height:   60)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/cb8c75fc1c7ccf73.jpg: Predicted in 67.379000 milli-seconds.
    license_plate: 45%	(left_x:  290   top_y:  468   width:  407   height:  125)
    license_plate: 47%	(left_x:  405   top_y:  475   width:  185   height:  124)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e5edf93b7c9f5edb.jpg: Predicted in 67.412000 milli-seconds.
    license_plate: 27%	(left_x:  336   top_y:  446   width:  164   height:   76)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/73467682c5995b65.jpg: Predicted in 67.634000 milli-seconds.
    license_plate: 30%	(left_x:  369   top_y:  485   width:  228   height:   47)
    license_plate: 37%	(left_x:  395   top_y:  485   width:  174   height:   29)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/ff69d090c735eccb.jpg: Predicted in 67.663000 milli-seconds.
    license_plate: 49%	(left_x:  524   top_y:  339   width:  259   height:  169)
    license_plate: 38%	(left_x:  542   top_y:  392   width:  222   height:   66)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a3ad91fabd188be3.jpg: Predicted in 67.275000 milli-seconds.
    license_plate: 65%	(left_x:  196   top_y:  414   width:  106   height:   52)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/1ca1155083156d72.jpg: Predicted in 67.249000 milli-seconds.
    license_plate: 48%	(left_x:  172   top_y:  354   width:  652   height:  165)
    license_plate: 91%	(left_x:  301   top_y:  406   width:  395   height:   56)
    license_plate: 62%	(left_x:  354   top_y:  363   width:  276   height:  151)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/52bf6b555e578a34.jpg: Predicted in 67.381000 milli-seconds.
    license_plate: 29%	(left_x:  970   top_y:  548   width:   47   height:   17)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/3e156de979eb0828.jpg: Predicted in 67.742000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/28e26a4eb646c67a.jpg: Predicted in 67.349000 milli-seconds.
    license_plate: 73%	(left_x:  374   top_y:  350   width:  308   height:   36)
    license_plate: 64%	(left_x:  403   top_y:  323   width:  255   height:   86)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0f0596b1c511e071.jpg: Predicted in 67.608000 milli-seconds.
    license_plate: 43%	(left_x:  874   top_y:  416   width:  167   height:  119)
    license_plate: 68%	(left_x:  906   top_y:  434   width:  114   height:   79)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/665cac5fd1cc2186.jpg: Predicted in 67.323000 milli-seconds.
    license_plate: 73%	(left_x:  121   top_y:  409   width:  103   height:   51)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4e2cb95b9c509b10.jpg: Predicted in 67.473000 milli-seconds.
    license_plate: 78%	(left_x:  435   top_y:  262   width:  250   height:   74)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/74ee6d1b58ae2e70.jpg: Predicted in 67.211000 milli-seconds.
    license_plate: 61%	(left_x:  450   top_y:  388   width:  141   height:   37)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/6b113b3edbadfe5d.jpg: Predicted in 67.030000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d8f6c135ec5486ff.jpg: Predicted in 67.432000 milli-seconds.
    license_plate: 66%	(left_x:  770   top_y:  425   width:  166   height:   72)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/09453a7c716a9ef3.jpg: Predicted in 67.462000 milli-seconds.
    license_plate: 67%	(left_x:  476   top_y:  632   width:  180   height:   57)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d8daed582e6cce2d.jpg: Predicted in 67.448000 milli-seconds.
    license_plate: 47%	(left_x:  454   top_y:  350   width:  323   height:  111)
    license_plate: 35%	(left_x:  519   top_y:  362   width:  185   height:   80)
    license_plate: 43%	(left_x:  578   top_y:  357   width:  166   height:   95)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/7f9740e95a74f2b9.jpg: Predicted in 67.246000 milli-seconds.
    license_plate: 33%	(left_x:  273   top_y:  365   width:  508   height:   85)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0d6ca8553971fefd.jpg: Predicted in 67.261000 milli-seconds.
    license_plate: 79%	(left_x:  232   top_y:  597   width:  142   height:   44)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/9498d0aff8f9ff8c.jpg: Predicted in 67.311000 milli-seconds.
    license_plate: 35%	(left_x:  843   top_y:  210   width:   67   height:   51)
    license_plate: 43%	(left_x:  945   top_y:  365   width:   53   height:   42)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/66771a4870c324f9.jpg: Predicted in 67.384000 milli-seconds.
    license_plate: 44%	(left_x:  744   top_y:  139   width:  181   height:   41)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5cc19b450a51ee4c.jpg: Predicted in 67.350000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/da1352ab6054a2e4.jpg: Predicted in 67.339000 milli-seconds.
    license_plate: 95%	(left_x:  -41   top_y:  378   width: 1105   height:  224)
    license_plate: 29%	(left_x:  173   top_y:  404   width:  567   height:  166)
    license_plate: 82%	(left_x:  265   top_y:  323   width:  440   height:  352)
    license_plate: 30%	(left_x:  311   top_y:  331   width:  233   height:  330)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/003a5aaf6d17c917.jpg: Predicted in 67.511000 milli-seconds.
    license_plate: 26%	(left_x:   89   top_y:   62   width:   74   height:   48)
    license_plate: 75%	(left_x:  212   top_y:  260   width:  149   height:   75)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/ccc1a2d44a290368.jpg: Predicted in 67.319000 milli-seconds.
    license_plate: 71%	(left_x:  488   top_y:  420   width:  195   height:   64)
    license_plate: 90%	(left_x:  528   top_y:  430   width:  102   height:   42)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/628a23a43f19c4db.jpg: Predicted in 67.527000 milli-seconds.
    license_plate: 59%	(left_x:  140   top_y:  373   width:  125   height:   62)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/315aead7766727b8.jpg: Predicted in 67.148000 milli-seconds.
    license_plate: 33%	(left_x:  333   top_y:  540   width:  121   height:   53)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/30b6cfb60bf44533.jpg: Predicted in 67.389000 milli-seconds.
    license_plate: 28%	(left_x:  778   top_y:  562   width:   40   height:   43)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4df1448703257ff0.jpg: Predicted in 67.312000 milli-seconds.
    license_plate: 51%	(left_x:  319   top_y:  658   width:  322   height:  149)
    license_plate: 55%	(left_x:  376   top_y:  691   width:  210   height:   83)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f3bd3adea149193b.jpg: Predicted in 67.161000 milli-seconds.
    license_plate: 73%	(left_x:  101   top_y:  542   width:  249   height:  107)
    license_plate: 59%	(left_x:  158   top_y:  559   width:  135   height:   95)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/647ff22cf6950211.jpg: Predicted in 67.389000 milli-seconds.
    license_plate: 61%	(left_x:  457   top_y:  470   width:  164   height:   76)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/128a59bca025bed6.jpg: Predicted in 67.502000 milli-seconds.
    license_plate: 25%	(left_x:  381   top_y:  407   width:  220   height:   44)
    license_plate: 80%	(left_x:  448   top_y:  401   width:  111   height:   47)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/12b7ea40074d3b20.jpg: Predicted in 67.324000 milli-seconds.
    license_plate: 35%	(left_x:  257   top_y:  535   width:  466   height:  143)
    license_plate: 46%	(left_x:  376   top_y:  571   width:  230   height:   56)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/7bcf6b4ec09f1ea7.jpg: Predicted in 67.257000 milli-seconds.
    license_plate: 70%	(left_x:  246   top_y:  350   width:  418   height:  157)
    license_plate: 85%	(left_x:  355   top_y:  385   width:  200   height:   85)
    license_plate: 64%	(left_x:  386   top_y:  406   width:  153   height:   52)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/911e694dcc0814bf.jpg: Predicted in 67.688000 milli-seconds.
    license_plate: 39%	(left_x:  113   top_y:  134   width:   31   height:    9)
    license_plate: 69%	(left_x:  801   top_y:  541   width:   82   height:   68)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2fe1f00b77a110a0.jpg: Predicted in 67.590000 milli-seconds.
    license_plate: 36%	(left_x:  427   top_y:  328   width:  149   height:   90)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/693acf98575cbe27.jpg: Predicted in 67.470000 milli-seconds.
    license_plate: 26%	(left_x:  152   top_y:  238   width:  112   height:   74)
    license_plate: 84%	(left_x:  178   top_y:  254   width:   84   height:   49)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0801961485534636.jpg: Predicted in 67.757000 milli-seconds.
    license_plate: 91%	(left_x:  372   top_y:  627   width:  240   height:  116)
    license_plate: 89%	(left_x:  405   top_y:  666   width:  182   height:   46)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/228bd52bbe043677.jpg: Predicted in 67.331000 milli-seconds.
    license_plate: 29%	(left_x:  497   top_y:  279   width:   82   height:   15)
    license_plate: 87%	(left_x:  509   top_y:  271   width:   66   height:   34)
    license_plate: 32%	(left_x:  906   top_y:  204   width:   53   height:   20)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b076ad266891d7aa.jpg: Predicted in 67.347000 milli-seconds.
    license_plate: 33%	(left_x:  934   top_y:  484   width:   40   height:   12)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b6ecda23586a6ba5.jpg: Predicted in 67.128000 milli-seconds.
    license_plate: 65%	(left_x:  289   top_y:  569   width:  145   height:   69)
    license_plate: 26%	(left_x:  535   top_y:  493   width:  218   height:   60)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/955a8e4c8ba8116e.jpg: Predicted in 67.433000 milli-seconds.
    license_plate: 26%	(left_x:  560   top_y:  442   width:  276   height:   84)
    license_plate: 82%	(left_x:  592   top_y:  395   width:  256   height:  180)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d53a3bc813477685.jpg: Predicted in 67.392000 milli-seconds.
    license_plate: 67%	(left_x:  405   top_y:  339   width:  118   height:   29)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/77e716224c85410f.jpg: Predicted in 67.218000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e6299b9e04680adb.jpg: Predicted in 67.152000 milli-seconds.
    license_plate: 41%	(left_x:  951   top_y:  237   width:   46   height:   13)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/1edfbe8c28a86caf.jpg: Predicted in 67.193000 milli-seconds.
    license_plate: 72%	(left_x:  310   top_y:  341   width:  370   height:   36)
    license_plate: 39%	(left_x:  421   top_y:  323   width:  144   height:   67)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/85f0b4db3eb512ce.jpg: Predicted in 67.004000 milli-seconds.
    license_plate: 64%	(left_x:  233   top_y:  363   width:  116   height:   48)
    license_plate: 65%	(left_x:  730   top_y:  337   width:  130   height:   42)
    license_plate: 45%	(left_x:  740   top_y:  347   width:   80   height:   24)
    license_plate: 32%	(left_x:  756   top_y:  342   width:   83   height:   30)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d28d71c2690c16ed.jpg: Predicted in 67.318000 milli-seconds.
    license_plate: 31%	(left_x:  301   top_y:   71   width:   53   height:   21)
    license_plate: 26%	(left_x:  585   top_y:  473   width:  218   height:   48)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/71881a9c0f7f0a86.jpg: Predicted in 67.453000 milli-seconds.
    license_plate: 26%	(left_x:  453   top_y:  456   width:  214   height:   78)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/82b53fe7f9147c96.jpg: Predicted in 67.408000 milli-seconds.
    license_plate: 45%	(left_x:  180   top_y:  389   width:   95   height:   34)
    license_plate: 46%	(left_x:  184   top_y:  380   width:  122   height:   54)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/25d5c2fbd8b99662.jpg: Predicted in 67.147000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f73b754cdb1ab677.jpg: Predicted in 67.471000 milli-seconds.
    license_plate: 84%	(left_x:  659   top_y:  491   width:  207   height:   83)
    license_plate: 37%	(left_x:  667   top_y:  517   width:  184   height:   34)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5581a1b7d2f0f2b4.jpg: Predicted in 67.332000 milli-seconds.
    license_plate: 90%	(left_x:  422   top_y:  612   width:  148   height:   83)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/fc0f46431cb1dbe9.jpg: Predicted in 67.254000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/1071b237587a698b.jpg: Predicted in 67.063000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/fa9147596edc058f.jpg: Predicted in 67.350000 milli-seconds.
    license_plate: 66%	(left_x:  624   top_y:  411   width:   77   height:   58)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b5e7183b6a5abe6c.jpg: Predicted in 67.180000 milli-seconds.
    license_plate: 55%	(left_x:  535   top_y:  523   width:  155   height:   64)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/67c834b73882a9f9.jpg: Predicted in 67.288000 milli-seconds.
    license_plate: 48%	(left_x:  526   top_y:  317   width:  414   height:  132)
    license_plate: 26%	(left_x:  599   top_y:  347   width:  240   height:   82)
    license_plate: 47%	(left_x:  645   top_y:  319   width:  177   height:  152)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/9286a99f243b359a.jpg: Predicted in 67.815000 milli-seconds.
    license_plate: 49%	(left_x:  787   top_y:  373   width:  104   height:   38)
    license_plate: 81%	(left_x:  797   top_y:  357   width:  103   height:   72)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/98bcfbc4d3c8abbc.jpg: Predicted in 67.248000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d648b2ae1abc09f3.jpg: Predicted in 67.422000 milli-seconds.
    license_plate: 73%	(left_x:  370   top_y:  519   width:  185   height:   74)
    license_plate: 41%	(left_x:  970   top_y:  399   width:   50   height:   26)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d2fe2b47668e9d8e.jpg: Predicted in 67.765000 milli-seconds.
    license_plate: 74%	(left_x:  827   top_y:  460   width:  210   height:  109)
    license_plate: 86%	(left_x:  863   top_y:  446   width:   84   height:  131)
    license_plate: 28%	(left_x:  871   top_y:  452   width:  117   height:   86)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2f90aaa72744452d.jpg: Predicted in 67.547000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/fb4e60b5cee8b88e.jpg: Predicted in 67.400000 milli-seconds.
    license_plate: 46%	(left_x:  278   top_y:  654   width:  170   height:  120)
    license_plate: 51%	(left_x:  294   top_y:  246   width:   66   height:   31)
    license_plate: 44%	(left_x:  306   top_y:  590   width:  122   height:  283)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0c756c9366a8cb10.jpg: Predicted in 67.342000 milli-seconds.
    license_plate: 78%	(left_x:  631   top_y:  569   width:  120   height:   88)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/064a8def3049d040.jpg: Predicted in 67.466000 milli-seconds.
    license_plate: 64%	(left_x:   47   top_y:  486   width:  185   height:  106)
    license_plate: 31%	(left_x:   80   top_y:  484   width:   80   height:  128)
    license_plate: 29%	(left_x:   82   top_y:  508   width:  174   height:   96)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/044417ca6134604f.jpg: Predicted in 67.198000 milli-seconds.
    license_plate: 72%	(left_x:  663   top_y:  345   width:  207   height:   65)
    license_plate: 61%	(left_x:  686   top_y:  355   width:  146   height:   39)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d5058059dc0ce0ad.jpg: Predicted in 67.173000 milli-seconds.
    license_plate: 32%	(left_x:  369   top_y:  386   width:  340   height:   78)
    license_plate: 43%	(left_x:  460   top_y:  365   width:  161   height:  134)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/bd12ea92bcab1afd.jpg: Predicted in 67.250000 milli-seconds.
    license_plate: 74%	(left_x:  713   top_y:  379   width:  119   height:   85)
    license_plate: 31%	(left_x:  741   top_y:  393   width:   73   height:   61)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/932eef14e5ee78b2.jpg: Predicted in 67.318000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/844adfe06e003c09.jpg: Predicted in 67.279000 milli-seconds.
    license_plate: 77%	(left_x:  589   top_y:  524   width:   60   height:   24)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5daafbcf76fa6602.jpg: Predicted in 67.687000 milli-seconds.
    license_plate: 54%	(left_x:  416   top_y:  514   width:  133   height:   37)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/efde2a25c1c9c924.jpg: Predicted in 67.705000 milli-seconds.
    license_plate: 47%	(left_x:  576   top_y:  433   width:  289   height:   49)
    license_plate: 45%	(left_x:  649   top_y:  423   width:  148   height:   66)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/abca9401dcf8f7ba.jpg: Predicted in 67.160000 milli-seconds.
    license_plate: 35%	(left_x:   62   top_y:  325   width:   57   height:   42)
    license_plate: 73%	(left_x:  443   top_y:  472   width:  154   height:   43)
    license_plate: 61%	(left_x:  467   top_y:  452   width:  113   height:   80)
    license_plate: 50%	(left_x:  683   top_y:  582   width:   64   height:   31)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d3729058c6a4186e.jpg: Predicted in 67.503000 milli-seconds.
    license_plate: 93%	(left_x:  566   top_y:  367   width:  166   height:   52)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2d6640fb230770fd.jpg: Predicted in 67.574000 milli-seconds.
    license_plate: 30%	(left_x:   40   top_y:  233   width:   22   height:   19)
    license_plate: 43%	(left_x:   98   top_y:  246   width:   54   height:   12)
    license_plate: 40%	(left_x:  108   top_y:  234   width:   39   height:   22)
    license_plate: 26%	(left_x:  640   top_y:  315   width:  228   height:   63)
    license_plate: 73%	(left_x:  687   top_y:  312   width:  149   height:   45)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e82f13b4a2fe69f3.jpg: Predicted in 67.244000 milli-seconds.
    license_plate: 48%	(left_x:  297   top_y:  391   width:  433   height:  348)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/71dbb47ebe504abe.jpg: Predicted in 67.507000 milli-seconds.
    license_plate: 92%	(left_x:  132   top_y:  366   width:   92   height:   34)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/299f0363ae21d1c3.jpg: Predicted in 67.166000 milli-seconds.
    license_plate: 28%	(left_x:  438   top_y:  404   width:  142   height:   77)
    license_plate: 77%	(left_x:  470   top_y:  388   width:  115   height:   81)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/fd07d2db70cf53d5.jpg: Predicted in 67.468000 milli-seconds.
    license_plate: 53%	(left_x:   91   top_y:  549   width:  156   height:   76)
    license_plate: 61%	(left_x:  101   top_y:  570   width:  151   height:   32)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d830c3573e57bfc0.jpg: Predicted in 67.548000 milli-seconds.
    license_plate: 29%	(left_x:  395   top_y:  587   width:  143   height:   26)
    license_plate: 80%	(left_x:  398   top_y:  578   width:  174   height:   47)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/08481c03daf6f35d.jpg: Predicted in 67.457000 milli-seconds.
    license_plate: 65%	(left_x:  357   top_y:  602   width:  337   height:   85)
    license_plate: 28%	(left_x:  384   top_y:  570   width:  213   height:  144)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/302d636c896c263f.jpg: Predicted in 67.506000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/9c92cadb0f6237c0.jpg: Predicted in 67.353000 milli-seconds.
    license_plate: 28%	(left_x:  742   top_y:  421   width:  264   height:   95)
    license_plate: 86%	(left_x:  813   top_y:  443   width:  125   height:   65)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b94993a2d67b455e.jpg: Predicted in 67.382000 milli-seconds.
    license_plate: 51%	(left_x:  292   top_y:  218   width:   94   height:   67)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5a122adbb1a776b7.jpg: Predicted in 67.508000 milli-seconds.
    license_plate: 71%	(left_x:  418   top_y:  422   width:  283   height:   50)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/ff85b09876d61631.jpg: Predicted in 67.620000 milli-seconds.
    license_plate: 38%	(left_x:  315   top_y:  638   width:  209   height:  182)
    license_plate: 41%	(left_x:  353   top_y:  682   width:  161   height:   76)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0e50ea14c4fc1353.jpg: Predicted in 67.318000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/cdb1e8621d1c624e.jpg: Predicted in 67.421000 milli-seconds.
    license_plate: 77%	(left_x:  652   top_y:  425   width:  224   height:   50)
    license_plate: 72%	(left_x:  687   top_y:  446   width:  158   height:   14)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5290f4027491d09b.jpg: Predicted in 67.413000 milli-seconds.
    license_plate: 61%	(left_x:  152   top_y:  549   width:  208   height:   70)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/6eb18d1ad17cd174.jpg: Predicted in 67.399000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e40cafe0f0d3a550.jpg: Predicted in 67.096000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/428a5131c8cff7da.jpg: Predicted in 67.630000 milli-seconds.
    license_plate: 75%	(left_x:  418   top_y:  526   width:  212   height:  116)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f0c7057612710f21.jpg: Predicted in 67.407000 milli-seconds.
    license_plate: 50%	(left_x:  377   top_y:  473   width:  231   height:   40)
    license_plate: 60%	(left_x:  403   top_y:  433   width:  185   height:   95)
    license_plate: 61%	(left_x:  449   top_y:  457   width:  108   height:   95)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/50c37aeaf19acd5b.jpg: Predicted in 67.177000 milli-seconds.
    license_plate: 51%	(left_x:  723   top_y:  480   width:   89   height:   54)
    license_plate: 39%	(left_x:  752   top_y:  457   width:  104   height:   66)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a9b2a2018f845393.jpg: Predicted in 67.657000 milli-seconds.
    license_plate: 68%	(left_x:   77   top_y:  376   width:  143   height:   51)
    license_plate: 85%	(left_x:  126   top_y:  353   width:   47   height:   97)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b1a50a3824887ee2.jpg: Predicted in 67.311000 milli-seconds.
    license_plate: 40%	(left_x:  323   top_y:  533   width:  259   height:   74)
    license_plate: 50%	(left_x:  388   top_y:  536   width:  189   height:   49)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/49595029959689b7.jpg: Predicted in 67.659000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/9238f5062ff1dc8d.jpg: Predicted in 67.657000 milli-seconds.
    license_plate: 34%	(left_x:  765   top_y:  489   width:  167   height:   57)
    license_plate: 67%	(left_x:  799   top_y:  500   width:   92   height:   31)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a72a8cda1bb31e34.jpg: Predicted in 67.289000 milli-seconds.
    license_plate: 49%	(left_x:  753   top_y:  568   width:  106   height:   75)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/daac3a6c64a79bea.jpg: Predicted in 67.180000 milli-seconds.
    license_plate: 32%	(left_x:   25   top_y:  365   width:  177   height:   40)
    license_plate: 34%	(left_x:   69   top_y:  349   width:   91   height:   64)
    license_plate: 34%	(left_x:   94   top_y:  342   width:   73   height:   85)
    license_plate: 58%	(left_x:  103   top_y:  366   width:   58   height:   35)
    license_plate: 31%	(left_x:  284   top_y:  881   width:  198   height:   36)
    license_plate: 30%	(left_x:  308   top_y:  867   width:  135   height:   64)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a929dc75c20da7d8.jpg: Predicted in 67.561000 milli-seconds.
    license_plate: 74%	(left_x:  269   top_y:  489   width:  283   height:  138)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/326c3286bcb5b2b0.jpg: Predicted in 67.519000 milli-seconds.
    license_plate: 63%	(left_x:  727   top_y:  438   width:  147   height:   56)
    license_plate: 58%	(left_x:  755   top_y:  438   width:   83   height:   47)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a52079c6cc4050cd.jpg: Predicted in 67.399000 milli-seconds.
    license_plate: 53%	(left_x:  440   top_y:  548   width:  131   height:   66)
    license_plate: 31%	(left_x:  447   top_y:  565   width:  115   height:   34)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f700c5b69c973db6.jpg: Predicted in 67.554000 milli-seconds.
    license_plate: 63%	(left_x:  327   top_y:  335   width:  121   height:   32)
    license_plate: 76%	(left_x:  348   top_y:  345   width:   65   height:   30)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b32671b7cc279583.jpg: Predicted in 67.346000 milli-seconds.
    license_plate: 32%	(left_x:  120   top_y:  444   width:  350   height:  117)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f9f539977bfea25e.jpg: Predicted in 67.290000 milli-seconds.
    license_plate: 42%	(left_x:  396   top_y:  477   width:  213   height:   89)
    license_plate: 57%	(left_x:  434   top_y:  493   width:  132   height:   58)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/29f7991e696e6e3f.jpg: Predicted in 67.246000 milli-seconds.
    license_plate: 56%	(left_x:  416   top_y:  308   width:  237   height:   82)
    license_plate: 78%	(left_x:  433   top_y:  244   width:  187   height:  194)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f3c9af100984744f.jpg: Predicted in 67.391000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d8cc6d566d01054b.jpg: Predicted in 67.698000 milli-seconds.
    license_plate: 70%	(left_x:  249   top_y:  484   width:  504   height:  196)
    license_plate: 30%	(left_x:  427   top_y:  474   width:  144   height:  182)
    license_plate: 49%	(left_x:  831   top_y:  512   width:  172   height:  132)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0fbd1b85fc01d2ae.jpg: Predicted in 67.411000 milli-seconds.
    license_plate: 48%	(left_x:   71   top_y:  410   width:   76   height:   45)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/44afab29f5fa0abf.jpg: Predicted in 67.240000 milli-seconds.
    license_plate: 74%	(left_x:  317   top_y:  562   width:  192   height:   89)
    license_plate: 35%	(left_x:  910   top_y:  459   width:  115   height:   60)
    license_plate: 44%	(left_x:  936   top_y:  455   width:   77   height:   50)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/3241f09a8964ddfb.jpg: Predicted in 67.409000 milli-seconds.
    license_plate: 57%	(left_x:  337   top_y:  506   width:   58   height:   40)
    license_plate: 41%	(left_x:  877   top_y:  366   width:   34   height:   20)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/50631041f74001aa.jpg: Predicted in 67.188000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/548a985825bb991f.jpg: Predicted in 67.437000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/dec4608715b2d28d.jpg: Predicted in 67.389000 milli-seconds.
    license_plate: 58%	(left_x:  186   top_y:  429   width:  209   height:   57)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/3d98e3713783e82d.jpg: Predicted in 67.398000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d7b71c9fd144d58d.jpg: Predicted in 67.327000 milli-seconds.
    license_plate: 83%	(left_x:  659   top_y:  478   width:  126   height:   80)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/03b7b71e1ffcb7a8.jpg: Predicted in 67.549000 milli-seconds.
    license_plate: 43%	(left_x:    1   top_y:  321   width:   60   height:   54)
    license_plate: 31%	(left_x:   46   top_y:  292   width:  162   height:   79)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/ebaefbcb87bfa8a1.jpg: Predicted in 67.284000 milli-seconds.
    license_plate: 77%	(left_x:  277   top_y:  512   width:   98   height:   37)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e08424f3cd5f6dac.jpg: Predicted in 67.504000 milli-seconds.
    license_plate: 94%	(left_x:  260   top_y:  286   width:  448   height:  105)
    license_plate: 74%	(left_x:  359   top_y:  307   width:  264   height:   65)
    license_plate: 69%	(left_x:  405   top_y:  256   width:  191   height:  176)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b91c3aaba25bf914.jpg: Predicted in 67.376000 milli-seconds.
    license_plate: 25%	(left_x:  284   top_y:  478   width:  156   height:   32)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e57d38ae6a921518.jpg: Predicted in 67.416000 milli-seconds.
    license_plate: 50%	(left_x:  215   top_y:  422   width:  105   height:   84)
    license_plate: 93%	(left_x:  221   top_y:  432   width:   90   height:   47)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/643c3cc1a4e85db2.jpg: Predicted in 67.649000 milli-seconds.
    license_plate: 33%	(left_x:  124   top_y:  403   width:  104   height:   48)
    license_plate: 40%	(left_x:  144   top_y:  357   width:   73   height:   91)
    license_plate: 41%	(left_x:  153   top_y:  347   width:   52   height:  146)
    license_plate: 37%	(left_x:  246   top_y:  423   width:   76   height:   25)
    license_plate: 66%	(left_x:  457   top_y:  374   width:  138   height:   81)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/659dbf4f8c0fd29d.jpg: Predicted in 67.360000 milli-seconds.
    license_plate: 78%	(left_x:  571   top_y:  638   width:  222   height:   97)
    license_plate: 75%	(left_x:  633   top_y:  651   width:   99   height:   69)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/fed5c0275ff3a440.jpg: Predicted in 67.433000 milli-seconds.
    license_plate: 67%	(left_x:   67   top_y:  373   width:  145   height:   57)
    license_plate: 32%	(left_x:  938   top_y:  185   width:   75   height:   64)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b01d46f9911d558b.jpg: Predicted in 67.554000 milli-seconds.
    license_plate: 27%	(left_x:  298   top_y:  158   width:  400   height:   90)
    license_plate: 67%	(left_x:  402   top_y:  129   width:  226   height:  136)
    license_plate: 67%	(left_x:  638   top_y:  163   width:  258   height:   78)
    license_plate: 72%	(left_x:  654   top_y:  184   width:  224   height:   38)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/80e02a518ffe4cb2.jpg: Predicted in 67.420000 milli-seconds.
    license_plate: 46%	(left_x:  362   top_y:  381   width:  341   height:  176)
    license_plate: 31%	(left_x:  418   top_y:  443   width:  225   height:   71)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/1ead26febde18ce9.jpg: Predicted in 67.360000 milli-seconds.
    license_plate: 65%	(left_x:  274   top_y:  296   width:  594   height:   90)
    license_plate: 38%	(left_x:  376   top_y:  303   width:  319   height:   74)
    license_plate: 61%	(left_x:  382   top_y:  229   width:  352   height:  221)
    license_plate: 35%	(left_x:  457   top_y:  265   width:  130   height:  112)
    license_plate: 72%	(left_x:  531   top_y:  234   width:   80   height:  212)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d5f4069e6734ac06.jpg: Predicted in 67.313000 milli-seconds.
    license_plate: 87%	(left_x:  262   top_y:  394   width:   73   height:   54)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/65e080f9ac466664.jpg: Predicted in 67.731000 milli-seconds.
    license_plate: 64%	(left_x:  208   top_y:  252   width:   68   height:   36)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5ff91ccfc6f15d04.jpg: Predicted in 67.380000 milli-seconds.
    license_plate: 84%	(left_x:  308   top_y:  282   width:  208   height:   54)
    license_plate: 36%	(left_x:  354   top_y:  267   width:  126   height:   85)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/944281d0485869f0.jpg: Predicted in 67.385000 milli-seconds.
    license_plate: 72%	(left_x:  344   top_y:  396   width:  359   height:  175)
    license_plate: 74%	(left_x:  432   top_y:  462   width:  214   height:   45)
    license_plate: 88%	(left_x:  432   top_y:  416   width:  189   height:  144)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/1f34454c4c073e1b.jpg: Predicted in 67.682000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/36d7b8b3cca3b0f5.jpg: Predicted in 67.473000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/00723dac8201a83e.jpg: Predicted in 67.503000 milli-seconds.
    license_plate: 46%	(left_x:    5   top_y:  408   width:   49   height:   27)
    license_plate: 29%	(left_x:  799   top_y:  346   width:   37   height:   30)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/c1d8b110186e095a.jpg: Predicted in 67.403000 milli-seconds.
    license_plate: 88%	(left_x:  374   top_y:  395   width:  166   height:   55)
    license_plate: 32%	(left_x:  395   top_y:  402   width:  106   height:   40)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/85b1dafd26aa98df.jpg: Predicted in 67.890000 milli-seconds.
    license_plate: 42%	(left_x:   13   top_y:   65   width:   34   height:   10)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/54bd93e5a4de808a.jpg: Predicted in 67.640000 milli-seconds.
    license_plate: 86%	(left_x:   92   top_y:  389   width:  175   height:   83)
    license_plate: 45%	(left_x:  130   top_y:  417   width:   98   height:   55)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/749b4451492fc250.jpg: Predicted in 67.530000 milli-seconds.
    license_plate: 69%	(left_x:  204   top_y:  462   width:   97   height:   50)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/c50d184afad1a9dc.jpg: Predicted in 68.176000 milli-seconds.
    license_plate: 38%	(left_x:  285   top_y:  286   width:  166   height:   52)
    license_plate: 56%	(left_x:  291   top_y:  373   width:   97   height:   94)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/7275e72e1996267f.jpg: Predicted in 67.296000 milli-seconds.
    license_plate: 57%	(left_x:  426   top_y:  384   width:  338   height:   39)
    license_plate: 45%	(left_x:  472   top_y:  371   width:  267   height:   75)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/1f0e643b125f00ec.jpg: Predicted in 67.142000 milli-seconds.
    license_plate: 86%	(left_x:  677   top_y:  431   width:  189   height:   98)
    license_plate: 68%	(left_x:  712   top_y:  457   width:  116   height:   54)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/63d3df798bc8840f.jpg: Predicted in 67.303000 milli-seconds.
    license_plate: 48%	(left_x:  362   top_y:  667   width:  319   height:  108)
    license_plate: 75%	(left_x:  397   top_y:  604   width:  212   height:  267)
    license_plate: 29%	(left_x:  430   top_y:  670   width:  132   height:   92)
    license_plate: 35%	(left_x:  461   top_y:  587   width:  132   height:  225)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5fb888d79b551331.jpg: Predicted in 67.736000 milli-seconds.
    license_plate: 76%	(left_x:  401   top_y:  454   width:  397   height:  225)
    license_plate: 76%	(left_x:  501   top_y:  481   width:  223   height:  164)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/52ceb1fc30b413e5.jpg: Predicted in 67.715000 milli-seconds.
    license_plate: 29%	(left_x:  461   top_y:  364   width:   39   height:   53)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b09813a8742277c7.jpg: Predicted in 67.589000 milli-seconds.
    license_plate: 27%	(left_x:  390   top_y:  352   width:  204   height:   43)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/ce97f7bc90e97109.jpg: Predicted in 67.410000 milli-seconds.
    license_plate: 29%	(left_x:  961   top_y:  170   width:   65   height:   18)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/da8186b39f042cdf.jpg: Predicted in 67.356000 milli-seconds.
    license_plate: 71%	(left_x:  381   top_y:  420   width:  167   height:   26)
    license_plate: 53%	(left_x:  408   top_y:  386   width:  107   height:   79)
    license_plate: 25%	(left_x:  421   top_y:  404   width:  112   height:   45)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/be9fd0014b5a4f2a.jpg: Predicted in 67.555000 milli-seconds.
    license_plate: 75%	(left_x:  203   top_y:  327   width:  182   height:   67)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/86c92b0402fec141.jpg: Predicted in 67.430000 milli-seconds.
    license_plate: 77%	(left_x:  298   top_y:  360   width:  283   height:   48)
    license_plate: 41%	(left_x:  372   top_y:  336   width:  163   height:   92)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/9ebbaf4d1f8d6e7b.jpg: Predicted in 67.627000 milli-seconds.
    license_plate: 79%	(left_x:  355   top_y:  145   width:  284   height:  209)
    license_plate: 93%	(left_x:  384   top_y:  190   width:  213   height:  111)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/791d63b1c3499e4f.jpg: Predicted in 67.606000 milli-seconds.
    license_plate: 26%	(left_x:  701   top_y:  485   width:  134   height:   70)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b3b61da98e22cd4a.jpg: Predicted in 67.865000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/ef1bd7b718a3f6e8.jpg: Predicted in 67.225000 milli-seconds.
    license_plate: 25%	(left_x:  740   top_y:  228   width:   73   height:   29)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/87b4f1202cd06440.jpg: Predicted in 67.694000 milli-seconds.
    license_plate: 59%	(left_x:   71   top_y:  553   width:  118   height:   24)
    license_plate: 47%	(left_x:  783   top_y:  646   width:  184   height:   39)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0170ea8e1a33375a.jpg: Predicted in 67.490000 milli-seconds.
    license_plate: 66%	(left_x:  576   top_y:  463   width:  204   height:   43)
    license_plate: 49%	(left_x:  614   top_y:  449   width:  123   height:   70)
    license_plate: 26%	(left_x:  624   top_y:  443   width:   56   height:   76)
    license_plate: 40%	(left_x:  910   top_y:  126   width:   52   height:   21)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a5d6f8d36672fa64.jpg: Predicted in 67.550000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e9fdd0a41cb61da6.jpg: Predicted in 67.855000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/19eba8ac64eed194.jpg: Predicted in 67.332000 milli-seconds.
    license_plate: 30%	(left_x:  242   top_y:  299   width:  131   height:   91)
    license_plate: 82%	(left_x:  270   top_y:  308   width:   69   height:   66)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/453a77009b27e253.jpg: Predicted in 67.538000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/46a7991da6461831.jpg: Predicted in 67.399000 milli-seconds.
    license_plate: 45%	(left_x:   71   top_y:  347   width:   75   height:   53)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e8b36c888a75d742.jpg: Predicted in 67.669000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b9f5b9acf1777acf.jpg: Predicted in 67.627000 milli-seconds.
    license_plate: 30%	(left_x:  197   top_y:  355   width:  552   height:  112)
    license_plate: 35%	(left_x:  334   top_y:  373   width:  289   height:   70)
    license_plate: 38%	(left_x:  407   top_y:  367   width:  137   height:   98)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/54ebca2064066a49.jpg: Predicted in 67.521000 milli-seconds.
    license_plate: 44%	(left_x:  472   top_y:  372   width:  146   height:   81)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/7b4e86e1c94d65de.jpg: Predicted in 67.197000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f194eaf4f3d1d835.jpg: Predicted in 67.341000 milli-seconds.
    license_plate: 52%	(left_x:  291   top_y:  484   width:   88   height:   60)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/85ddb7e0f60156a3.jpg: Predicted in 67.715000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/64de505bd2bac82b.jpg: Predicted in 67.580000 milli-seconds.
    license_plate: 43%	(left_x:  186   top_y:  564   width:  228   height:   79)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/70e7b5173fe1e289.jpg: Predicted in 67.695000 milli-seconds.
    license_plate: 60%	(left_x:  100   top_y:  248   width:  390   height:   91)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/ee047c8ca2cca8a2.jpg: Predicted in 67.460000 milli-seconds.
    license_plate: 81%	(left_x:  325   top_y:  332   width:  409   height:   89)
    license_plate: 47%	(left_x:  450   top_y:  363   width:  139   height:   30)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/091c033b2a7df15b.jpg: Predicted in 67.464000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e52d4f347b2cf90f.jpg: Predicted in 67.292000 milli-seconds.
    license_plate: 56%	(left_x:  480   top_y:  414   width:  130   height:   99)
    license_plate: 47%	(left_x:  506   top_y:  442   width:  102   height:   46)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5df8816356fc2b29.jpg: Predicted in 67.546000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/28bd6c6b1050e055.jpg: Predicted in 67.400000 milli-seconds.
    license_plate: 30%	(left_x:  146   top_y:  236   width:   85   height:   99)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/460753acbd6e6dad.jpg: Predicted in 67.562000 milli-seconds.
    license_plate: 66%	(left_x:  302   top_y:  558   width:  133   height:   32)
    license_plate: 63%	(left_x:  308   top_y:  524   width:  118   height:   90)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/6a0ef049e5ec4b16.jpg: Predicted in 67.527000 milli-seconds.
    license_plate: 65%	(left_x:   16   top_y:  449   width:  234   height:   78)
    license_plate: 72%	(left_x:   43   top_y:  389   width:  184   height:  190)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b7cdb5c0c99002b7.jpg: Predicted in 67.330000 milli-seconds.
    license_plate: 91%	(left_x:  709   top_y:  278   width:  188   height:   50)
    license_plate: 27%	(left_x:  722   top_y:  252   width:  163   height:  127)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/61a69ad713142e45.jpg: Predicted in 67.466000 milli-seconds.
    license_plate: 66%	(left_x:  303   top_y:  484   width:  120   height:   80)
    license_plate: 83%	(left_x:  308   top_y:  510   width:  117   height:   31)
    license_plate: 72%	(left_x:  329   top_y:  505   width:   80   height:   60)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/9995dfc6e0eae5a1.jpg: Predicted in 67.491000 milli-seconds.
    license_plate: 50%	(left_x:  393   top_y:  389   width:  133   height:   63)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/aac1aaaf824b63dd.jpg: Predicted in 67.332000 milli-seconds.
    license_plate: 26%	(left_x:   20   top_y:  238   width:   95   height:   73)
    license_plate: 56%	(left_x:   47   top_y:  261   width:   42   height:   26)
    license_plate: 67%	(left_x:  444   top_y:  436   width:  157   height:   56)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0ee91c4938b6e7ee.jpg: Predicted in 67.529000 milli-seconds.
    license_plate: 62%	(left_x:  296   top_y:  565   width:  152   height:   75)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/18098dc9ee6aad80.jpg: Predicted in 67.426000 milli-seconds.
    license_plate: 52%	(left_x:  640   top_y:  389   width:  114   height:   33)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d27e094e98374ff8.jpg: Predicted in 67.539000 milli-seconds.
    license_plate: 74%	(left_x:  537   top_y:  439   width:  218   height:   47)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/fa897478280a2758.jpg: Predicted in 67.393000 milli-seconds.
    license_plate: 33%	(left_x:  506   top_y:  476   width:  223   height:   69)
    license_plate: 31%	(left_x:  555   top_y:  463   width:  128   height:   62)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/1a3e9d87d733ff7f.jpg: Predicted in 67.722000 milli-seconds.
    license_plate: 67%	(left_x:  827   top_y:  336   width:   94   height:   36)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b6580dec5ada277d.jpg: Predicted in 67.386000 milli-seconds.
    license_plate: 54%	(left_x:  218   top_y:  351   width:  235   height:   52)
    license_plate: 76%	(left_x:  226   top_y:  282   width:  206   height:  181)
    license_plate: 40%	(left_x:  243   top_y:  320   width:  165   height:   90)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/59b83c5f8f1dcfa4.jpg: Predicted in 67.487000 milli-seconds.
    license_plate: 29%	(left_x:  271   top_y:  727   width:  248   height:   82)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/140e8d10ff02e7e7.jpg: Predicted in 67.174000 milli-seconds.
    license_plate: 39%	(left_x:  438   top_y:  395   width:  174   height:   37)
    license_plate: 42%	(left_x:  484   top_y:  397   width:   78   height:   31)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/1e6e75488d7878aa.jpg: Predicted in 67.281000 milli-seconds.
    license_plate: 35%	(left_x:  697   top_y:  364   width:   42   height:   24)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/bbcac63e32bd8137.jpg: Predicted in 67.558000 milli-seconds.
    license_plate: 52%	(left_x:  183   top_y:  221   width:  784   height:  335)
    license_plate: 49%	(left_x:  405   top_y:  230   width:  267   height:  250)
    license_plate: 55%	(left_x:  428   top_y:  300   width:  292   height:  155)
    license_plate: 30%	(left_x:  504   top_y:  289   width:  180   height:  134)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/334f31e809ec9a77.jpg: Predicted in 67.398000 milli-seconds.
    license_plate: 28%	(left_x:  139   top_y:  439   width:  143   height:   49)
    license_plate: 37%	(left_x:  150   top_y:  434   width:   78   height:   66)
    license_plate: 43%	(left_x:  165   top_y:  447   width:   79   height:   41)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/be654a7eabe0e891.jpg: Predicted in 67.569000 milli-seconds.
    license_plate: 67%	(left_x:  790   top_y:  374   width:  130   height:   63)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/575d96d014e4fbab.jpg: Predicted in 67.484000 milli-seconds.
    license_plate: 80%	(left_x:  376   top_y:  480   width:  151   height:   57)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/895d440e05b2a8d8.jpg: Predicted in 67.518000 milli-seconds.
    license_plate: 25%	(left_x:  198   top_y:  516   width:  202   height:   86)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/7539902687e8f3a4.jpg: Predicted in 67.310000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e1925cd0ae28fbd9.jpg: Predicted in 67.767000 milli-seconds.
    license_plate: 85%	(left_x:  435   top_y:  376   width:  168   height:   77)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/185fd6653258d3ed.jpg: Predicted in 67.509000 milli-seconds.
    license_plate: 48%	(left_x:  637   top_y:  420   width:  193   height:   69)
    license_plate: 42%	(left_x:  691   top_y:  443   width:   91   height:   21)
    license_plate: 76%	(left_x:  699   top_y:  430   width:   81   height:   44)
    license_plate: 38%	(left_x:  702   top_y:  494   width:   74   height:   26)
    license_plate: 41%	(left_x:  719   top_y:  499   width:   38   height:   16)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/26183d88c4f8012f.jpg: Predicted in 67.327000 milli-seconds.
    license_plate: 63%	(left_x:  194   top_y:  442   width:  194   height:   96)
    license_plate: 68%	(left_x:  229   top_y:  470   width:  135   height:   37)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/288a95c01e43cd14.jpg: Predicted in 67.525000 milli-seconds.
    license_plate: 76%	(left_x:  465   top_y:  378   width:  197   height:   48)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/16789af24af158d8.jpg: Predicted in 67.820000 milli-seconds.
    license_plate: 40%	(left_x:  232   top_y:  473   width:  441   height:  100)
    license_plate: 64%	(left_x:  290   top_y:  493   width:  326   height:   60)
    license_plate: 28%	(left_x:  353   top_y:  468   width:  171   height:  118)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/16c15b29b30148c2.jpg: Predicted in 67.574000 milli-seconds.
    license_plate: 60%	(left_x:  184   top_y:  582   width:  439   height:   94)
    license_plate: 28%	(left_x:  289   top_y:  595   width:  248   height:   63)
    license_plate: 83%	(left_x:  374   top_y:  573   width:   80   height:  133)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/488722909dd9c0ac.jpg: Predicted in 67.532000 milli-seconds.
    license_plate: 46%	(left_x:  326   top_y:  671   width:  173   height:   97)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b6e55f97085c3732.jpg: Predicted in 67.545000 milli-seconds.
    license_plate: 41%	(left_x:  195   top_y:  549   width:  118   height:   48)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/17585ef2efb5ef9b.jpg: Predicted in 67.390000 milli-seconds.
    license_plate: 29%	(left_x:  282   top_y:  777   width:  275   height:   56)
    license_plate: 44%	(left_x:  317   top_y:  764   width:  160   height:   76)
    license_plate: 35%	(left_x:  410   top_y:  656   width:   70   height:  325)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/3feecc5809a29627.jpg: Predicted in 67.592000 milli-seconds.
    license_plate: 41%	(left_x:  186   top_y:  702   width:  310   height:   62)
    license_plate: 43%	(left_x:  289   top_y:  703   width:  161   height:   60)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/bf2f25f3ed9ff4d5.jpg: Predicted in 67.196000 milli-seconds.
    license_plate: 28%	(left_x:  307   top_y:  413   width:  449   height:   82)
    license_plate: 76%	(left_x:  400   top_y:  417   width:  221   height:   81)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/343326e127297379.jpg: Predicted in 67.626000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/6846c275ded01f85.jpg: Predicted in 67.418000 milli-seconds.
    license_plate: 62%	(left_x:  531   top_y:  487   width:  403   height:  204)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/defd8f4b30b3e1e1.jpg: Predicted in 67.633000 milli-seconds.
    license_plate: 56%	(left_x:  182   top_y:  374   width:   95   height:  103)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a67bc3f5f8650c73.jpg: Predicted in 67.607000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/ea02d99ba51372ee.jpg: Predicted in 67.281000 milli-seconds.
    license_plate: 29%	(left_x:  144   top_y:  591   width:  153   height:   42)
    license_plate: 51%	(left_x:  819   top_y:  322   width:   33   height:   15)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/5373ec295ea62c47.jpg: Predicted in 67.368000 milli-seconds.
    license_plate: 71%	(left_x:   27   top_y:  204   width:  118   height:   28)
    license_plate: 54%	(left_x:  809   top_y:  377   width:  173   height:   68)
    license_plate: 82%	(left_x:  840   top_y:  383   width:  103   height:   53)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/cb4b754537798d23.jpg: Predicted in 67.521000 milli-seconds.
    license_plate: 48%	(left_x:    4   top_y:  344   width:   43   height:   33)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/125622a2508e3b1d.jpg: Predicted in 67.682000 milli-seconds.
    license_plate: 25%	(left_x:   35   top_y:  435   width:   33   height:   23)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/6306fc7f3573eb70.jpg: Predicted in 67.689000 milli-seconds.
    license_plate: 44%	(left_x:  528   top_y:  437   width:   25   height:    9)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/43b1b0028dd2db8d.jpg: Predicted in 67.418000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a56e82026bb71ad5.jpg: Predicted in 67.713000 milli-seconds.
    license_plate: 71%	(left_x:  376   top_y:  386   width:  293   height:  139)
    license_plate: 90%	(left_x:  426   top_y:  424   width:  193   height:   61)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a21aa7c1d7b93d12.jpg: Predicted in 67.468000 milli-seconds.
    license_plate: 38%	(left_x:  115   top_y:  393   width:  278   height:   90)
    license_plate: 53%	(left_x:  213   top_y:  372   width:   96   height:  117)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2e8a83cfb1afe7ba.jpg: Predicted in 67.805000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/246a0e4264a39433.jpg: Predicted in 67.455000 milli-seconds.
    license_plate: 51%	(left_x:  282   top_y:  447   width:  155   height:   61)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/37ff77d13a54aac2.jpg: Predicted in 67.581000 milli-seconds.
    license_plate: 63%	(left_x:  381   top_y:  466   width:  138   height:   38)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/9a4fe6361597f264.jpg: Predicted in 67.473000 milli-seconds.
    license_plate: 61%	(left_x:  361   top_y:  424   width:  267   height:   69)
    license_plate: 29%	(left_x:  453   top_y:  410   width:  114   height:   92)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/e1120bfe48f85c1b.jpg: Predicted in 67.622000 milli-seconds.
    license_plate: 72%	(left_x:  186   top_y:  446   width:  124   height:   77)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/8ceed1ec9bc3ac45.jpg: Predicted in 67.360000 milli-seconds.
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2c09fab0221652c5.jpg: Predicted in 67.285000 milli-seconds.
    license_plate: 27%	(left_x:  488   top_y:  402   width:  181   height:   68)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/39c193bcc0c00f9e.jpg: Predicted in 67.259000 milli-seconds.
    license_plate: 53%	(left_x:  187   top_y:  369   width:  295   height:  138)
    license_plate: 32%	(left_x:  274   top_y:  409   width:  144   height:   44)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/4c04b488ddc48225.jpg: Predicted in 67.429000 milli-seconds.
    license_plate: 90%	(left_x:  757   top_y:  332   width:  173   height:   60)
    license_plate: 78%	(left_x:  783   top_y:  345   width:  114   height:   38)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/2753536137929411.jpg: Predicted in 67.402000 milli-seconds.
    license_plate: 56%	(left_x:  109   top_y:  397   width:  149   height:   52)
    license_plate: 90%	(left_x:  140   top_y:  403   width:   80   height:   57)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/a54e7bb80448f3b0.jpg: Predicted in 67.516000 milli-seconds.
    license_plate: 27%	(left_x:    5   top_y:  412   width:   48   height:  108)
    license_plate: 53%	(left_x:  683   top_y:  351   width:  183   height:   77)
    license_plate: 33%	(left_x:  721   top_y:  366   width:   79   height:   47)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/d59b7a0ff294e3be.jpg: Predicted in 67.384000 milli-seconds.
    license_plate: 84%	(left_x:  421   top_y:  569   width:  164   height:   54)
    license_plate: 57%	(left_x:  453   top_y:  581   width:   88   height:   24)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/b1096bc91a89b0cf.jpg: Predicted in 67.376000 milli-seconds.
    license_plate: 78%	(left_x:  692   top_y:  388   width:  144   height:   58)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/edecf3a6d569b7c9.jpg: Predicted in 67.520000 milli-seconds.
    license_plate: 34%	(left_x:  -77   top_y:  780   width:  373   height:  132)
    license_plate: 59%	(left_x:  -27   top_y:  727   width:  244   height:  234)
    license_plate: 43%	(left_x:   27   top_y:  776   width:  125   height:  146)
    license_plate: 39%	(left_x:   29   top_y:  771   width:  213   height:  172)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/321ffb38656f7311.jpg: Predicted in 67.740000 milli-seconds.
    license_plate: 49%	(left_x:  169   top_y:  431   width:   31   height:    9)
    license_plate: 26%	(left_x:  170   top_y:  455   width:   29   height:   17)
    license_plate: 45%	(left_x:  466   top_y:  437   width:   72   height:   24)
    license_plate: 50%	(left_x:  622   top_y:  503   width:  128   height:   30)
    license_plate: 69%	(left_x:  646   top_y:  490   width:   71   height:   50)
    license_plate: 32%	(left_x:  804   top_y:  416   width:  101   height:   41)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/33e0333ff031c4fb.jpg: Predicted in 67.500000 milli-seconds.
    license_plate: 58%	(left_x:  756   top_y:  672   width:  103   height:  109)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0f4bfc46402a9f52.jpg: Predicted in 67.282000 milli-seconds.
    license_plate: 42%	(left_x:  516   top_y:  283   width:  299   height:  295)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/f6256bceaf66693f.jpg: Predicted in 67.713000 milli-seconds.
    license_plate: 51%	(left_x:  520   top_y:  566   width:  157   height:   84)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/0727983dd5f9e4e6.jpg: Predicted in 67.509000 milli-seconds.
    license_plate: 47%	(left_x:  471   top_y:  587   width:  212   height:   98)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/12d3b6665f7e1cf6.jpg: Predicted in 67.657000 milli-seconds.
    license_plate: 47%	(left_x:    1   top_y:  358   width:   73   height:   50)
    license_plate: 64%	(left_x:   16   top_y:  329   width:   56   height:  100)
    Enter Image Path:  Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    data/test/c91ee912164d8ecb.jpg: Predicted in 67.583000 milli-seconds.
    license_plate: 29%	(left_x:  389   top_y:  314   width:  287   height:   69)
    Enter Image Path: 

# Metrics
**Use -map flag while training for charts**  
mAP-chart (red-line) and Loss-chart (blue-line) will be saved in root directory.  
mAP will be calculated for each 4 Epochs ~ 240 batches

**I lost my chart due to intruption, but the records are available in training outputs**

## Comparing the weights on validation set

## for yolov4-obj_last.weights (600 iterations) ~ 2hours


```python
!./darknet detector map data/obj.data cfg/yolov4-obj.cfg backup/yolov4-obj_last.weights
```

     CUDA-version: 11010 (11020), cuDNN: 7.6.5, CUDNN_HALF=1, GPU count: 1  
     CUDNN_HALF=1 
     OpenCV version: 3.2.0
     0 : compute_capability = 370, cudnn_half = 0, GPU: Tesla K80 
    net.optimized_memory = 0 
    mini_batch = 1, batch = 1, time_steps = 1, train = 0 
       layer   filters  size/strd(dil)      input                output
       0 Create CUDA-stream - 0 
     Create cudnn-handle 0 
    conv     32       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  32 0.299 BF
       1 conv     64       3 x 3/ 2    416 x 416 x  32 ->  208 x 208 x  64 1.595 BF
       2 conv     64       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  64 0.354 BF
       3 route  1 		                           ->  208 x 208 x  64 
       4 conv     64       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  64 0.354 BF
       5 conv     32       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  32 0.177 BF
       6 conv     64       3 x 3/ 1    208 x 208 x  32 ->  208 x 208 x  64 1.595 BF
       7 Shortcut Layer: 4,  wt = 0, wn = 0, outputs: 208 x 208 x  64 0.003 BF
       8 conv     64       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  64 0.354 BF
       9 route  8 2 	                           ->  208 x 208 x 128 
      10 conv     64       1 x 1/ 1    208 x 208 x 128 ->  208 x 208 x  64 0.709 BF
      11 conv    128       3 x 3/ 2    208 x 208 x  64 ->  104 x 104 x 128 1.595 BF
      12 conv     64       1 x 1/ 1    104 x 104 x 128 ->  104 x 104 x  64 0.177 BF
      13 route  11 		                           ->  104 x 104 x 128 
      14 conv     64       1 x 1/ 1    104 x 104 x 128 ->  104 x 104 x  64 0.177 BF
      15 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
      16 conv     64       3 x 3/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.797 BF
      17 Shortcut Layer: 14,  wt = 0, wn = 0, outputs: 104 x 104 x  64 0.001 BF
      18 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
      19 conv     64       3 x 3/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.797 BF
      20 Shortcut Layer: 17,  wt = 0, wn = 0, outputs: 104 x 104 x  64 0.001 BF
      21 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
      22 route  21 12 	                           ->  104 x 104 x 128 
      23 conv    128       1 x 1/ 1    104 x 104 x 128 ->  104 x 104 x 128 0.354 BF
      24 conv    256       3 x 3/ 2    104 x 104 x 128 ->   52 x  52 x 256 1.595 BF
      25 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
      26 route  24 		                           ->   52 x  52 x 256 
      27 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
      28 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      29 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      30 Shortcut Layer: 27,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      31 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      32 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      33 Shortcut Layer: 30,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      34 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      35 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      36 Shortcut Layer: 33,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      37 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      38 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      39 Shortcut Layer: 36,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      40 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      41 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      42 Shortcut Layer: 39,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      43 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      44 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      45 Shortcut Layer: 42,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      46 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      47 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      48 Shortcut Layer: 45,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      49 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      50 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      51 Shortcut Layer: 48,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      52 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      53 route  52 25 	                           ->   52 x  52 x 256 
      54 conv    256       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 256 0.354 BF
      55 conv    512       3 x 3/ 2     52 x  52 x 256 ->   26 x  26 x 512 1.595 BF
      56 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
      57 route  55 		                           ->   26 x  26 x 512 
      58 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
      59 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      60 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      61 Shortcut Layer: 58,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      62 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      63 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      64 Shortcut Layer: 61,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      65 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      66 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      67 Shortcut Layer: 64,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      68 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      69 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      70 Shortcut Layer: 67,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      71 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      72 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      73 Shortcut Layer: 70,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      74 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      75 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      76 Shortcut Layer: 73,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      77 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      78 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      79 Shortcut Layer: 76,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      80 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      81 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      82 Shortcut Layer: 79,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      83 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      84 route  83 56 	                           ->   26 x  26 x 512 
      85 conv    512       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 512 0.354 BF
      86 conv   1024       3 x 3/ 2     26 x  26 x 512 ->   13 x  13 x1024 1.595 BF
      87 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
      88 route  86 		                           ->   13 x  13 x1024 
      89 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
      90 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
      91 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      92 Shortcut Layer: 89,  wt = 0, wn = 0, outputs:  13 x  13 x 512 0.000 BF
      93 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
      94 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      95 Shortcut Layer: 92,  wt = 0, wn = 0, outputs:  13 x  13 x 512 0.000 BF
      96 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
      97 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      98 Shortcut Layer: 95,  wt = 0, wn = 0, outputs:  13 x  13 x 512 0.000 BF
      99 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
     100 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
     101 Shortcut Layer: 98,  wt = 0, wn = 0, outputs:  13 x  13 x 512 0.000 BF
     102 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
     103 route  102 87 	                           ->   13 x  13 x1024 
     104 conv   1024       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x1024 0.354 BF
     105 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     106 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     107 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     108 max                5x 5/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.002 BF
     109 route  107 		                           ->   13 x  13 x 512 
     110 max                9x 9/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.007 BF
     111 route  107 		                           ->   13 x  13 x 512 
     112 max               13x13/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.015 BF
     113 route  112 110 108 107 	                   ->   13 x  13 x2048 
     114 conv    512       1 x 1/ 1     13 x  13 x2048 ->   13 x  13 x 512 0.354 BF
     115 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     116 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     117 conv    256       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 256 0.044 BF
     118 upsample                 2x    13 x  13 x 256 ->   26 x  26 x 256
     119 route  85 		                           ->   26 x  26 x 512 
     120 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     121 route  120 118 	                           ->   26 x  26 x 512 
     122 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     123 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     124 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     125 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     126 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     127 conv    128       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 128 0.044 BF
     128 upsample                 2x    26 x  26 x 128 ->   52 x  52 x 128
     129 route  54 		                           ->   52 x  52 x 256 
     130 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
     131 route  130 128 	                           ->   52 x  52 x 256 
     132 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
     133 conv    256       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 256 1.595 BF
     134 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
     135 conv    256       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 256 1.595 BF
     136 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
     137 conv    256       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 256 1.595 BF
     138 conv     18       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x  18 0.025 BF
     139 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.20
    nms_kind: greedynms (1), beta = 0.600000 
     140 route  136 		                           ->   52 x  52 x 128 
     141 conv    256       3 x 3/ 2     52 x  52 x 128 ->   26 x  26 x 256 0.399 BF
     142 route  141 126 	                           ->   26 x  26 x 512 
     143 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     144 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     145 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     146 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     147 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     148 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     149 conv     18       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x  18 0.012 BF
     150 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.10
    nms_kind: greedynms (1), beta = 0.600000 
     151 route  147 		                           ->   26 x  26 x 256 
     152 conv    512       3 x 3/ 2     26 x  26 x 256 ->   13 x  13 x 512 0.399 BF
     153 route  152 116 	                           ->   13 x  13 x1024 
     154 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     155 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     156 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     157 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     158 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     159 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     160 conv     18       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x  18 0.006 BF
     161 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
    nms_kind: greedynms (1), beta = 0.600000 
    Total BFLOPS 59.563 
    avg_outputs = 489778 
     Allocate additional workspace_size = 12.46 MB 
    Loading weights from backup/yolov4-obj_last.weights...
     seen 64, trained: 38 K-images (0 Kilo-batches_64) 
    Done! Loaded 162 layers from weights-file 
    
     calculation mAP (mean average precision)...
     Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    388
     detections_count = 12327, unique_truth_count = 512  
    class_id = 0, name = license_plate, ap = 46.27%   	 (TP = 311, FP = 313) 
    
     for conf_thresh = 0.25, precision = 0.50, recall = 0.61, F1-score = 0.55 
     for conf_thresh = 0.25, TP = 311, FP = 313, FN = 201, average IoU = 33.41 % 
    
     IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
     mean average precision (mAP@0.50) = 0.462668, or 46.27 % 
    Total Detection Time: 209 Seconds
    
    Set -points flag:
     `-points 101` for MS COCO 
     `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data) 
     `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom dataset
    

*   Precision: 50 %
*   Average Precision: 46.27 %
*   Recall: 61 %
*   F1-score: 55 %
*   Average IoU: 33.41 %
*   mAP@0.5: 46.27 %
*   Confusion Matrix:
    *   TP = 311
    *   FP = 313
    *   FN = 201
    *   unique_truth_count (TP+FN) = 512
    *   detections_count = 12327





### For best-weight.weights ~ 16hours


```python
!./darknet detector map data/obj.data cfg/yolov4-obj.cfg backup/custom.weights
```

     CUDA-version: 11010 (11020), cuDNN: 7.6.5, CUDNN_HALF=1, GPU count: 1  
     CUDNN_HALF=1 
     OpenCV version: 3.2.0
     0 : compute_capability = 370, cudnn_half = 0, GPU: Tesla K80 
    net.optimized_memory = 0 
    mini_batch = 1, batch = 1, time_steps = 1, train = 0 
       layer   filters  size/strd(dil)      input                output
       0 Create CUDA-stream - 0 
     Create cudnn-handle 0 
    conv     32       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  32 0.299 BF
       1 conv     64       3 x 3/ 2    416 x 416 x  32 ->  208 x 208 x  64 1.595 BF
       2 conv     64       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  64 0.354 BF
       3 route  1 		                           ->  208 x 208 x  64 
       4 conv     64       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  64 0.354 BF
       5 conv     32       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  32 0.177 BF
       6 conv     64       3 x 3/ 1    208 x 208 x  32 ->  208 x 208 x  64 1.595 BF
       7 Shortcut Layer: 4,  wt = 0, wn = 0, outputs: 208 x 208 x  64 0.003 BF
       8 conv     64       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  64 0.354 BF
       9 route  8 2 	                           ->  208 x 208 x 128 
      10 conv     64       1 x 1/ 1    208 x 208 x 128 ->  208 x 208 x  64 0.709 BF
      11 conv    128       3 x 3/ 2    208 x 208 x  64 ->  104 x 104 x 128 1.595 BF
      12 conv     64       1 x 1/ 1    104 x 104 x 128 ->  104 x 104 x  64 0.177 BF
      13 route  11 		                           ->  104 x 104 x 128 
      14 conv     64       1 x 1/ 1    104 x 104 x 128 ->  104 x 104 x  64 0.177 BF
      15 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
      16 conv     64       3 x 3/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.797 BF
      17 Shortcut Layer: 14,  wt = 0, wn = 0, outputs: 104 x 104 x  64 0.001 BF
      18 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
      19 conv     64       3 x 3/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.797 BF
      20 Shortcut Layer: 17,  wt = 0, wn = 0, outputs: 104 x 104 x  64 0.001 BF
      21 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
      22 route  21 12 	                           ->  104 x 104 x 128 
      23 conv    128       1 x 1/ 1    104 x 104 x 128 ->  104 x 104 x 128 0.354 BF
      24 conv    256       3 x 3/ 2    104 x 104 x 128 ->   52 x  52 x 256 1.595 BF
      25 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
      26 route  24 		                           ->   52 x  52 x 256 
      27 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
      28 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      29 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      30 Shortcut Layer: 27,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      31 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      32 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      33 Shortcut Layer: 30,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      34 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      35 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      36 Shortcut Layer: 33,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      37 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      38 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      39 Shortcut Layer: 36,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      40 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      41 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      42 Shortcut Layer: 39,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      43 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      44 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      45 Shortcut Layer: 42,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      46 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      47 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      48 Shortcut Layer: 45,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      49 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      50 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      51 Shortcut Layer: 48,  wt = 0, wn = 0, outputs:  52 x  52 x 128 0.000 BF
      52 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      53 route  52 25 	                           ->   52 x  52 x 256 
      54 conv    256       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 256 0.354 BF
      55 conv    512       3 x 3/ 2     52 x  52 x 256 ->   26 x  26 x 512 1.595 BF
      56 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
      57 route  55 		                           ->   26 x  26 x 512 
      58 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
      59 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      60 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      61 Shortcut Layer: 58,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      62 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      63 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      64 Shortcut Layer: 61,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      65 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      66 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      67 Shortcut Layer: 64,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      68 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      69 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      70 Shortcut Layer: 67,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      71 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      72 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      73 Shortcut Layer: 70,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      74 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      75 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      76 Shortcut Layer: 73,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      77 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      78 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      79 Shortcut Layer: 76,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      80 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      81 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      82 Shortcut Layer: 79,  wt = 0, wn = 0, outputs:  26 x  26 x 256 0.000 BF
      83 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      84 route  83 56 	                           ->   26 x  26 x 512 
      85 conv    512       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 512 0.354 BF
      86 conv   1024       3 x 3/ 2     26 x  26 x 512 ->   13 x  13 x1024 1.595 BF
      87 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
      88 route  86 		                           ->   13 x  13 x1024 
      89 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
      90 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
      91 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      92 Shortcut Layer: 89,  wt = 0, wn = 0, outputs:  13 x  13 x 512 0.000 BF
      93 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
      94 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      95 Shortcut Layer: 92,  wt = 0, wn = 0, outputs:  13 x  13 x 512 0.000 BF
      96 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
      97 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      98 Shortcut Layer: 95,  wt = 0, wn = 0, outputs:  13 x  13 x 512 0.000 BF
      99 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
     100 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
     101 Shortcut Layer: 98,  wt = 0, wn = 0, outputs:  13 x  13 x 512 0.000 BF
     102 conv    512       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.089 BF
     103 route  102 87 	                           ->   13 x  13 x1024 
     104 conv   1024       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x1024 0.354 BF
     105 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     106 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     107 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     108 max                5x 5/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.002 BF
     109 route  107 		                           ->   13 x  13 x 512 
     110 max                9x 9/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.007 BF
     111 route  107 		                           ->   13 x  13 x 512 
     112 max               13x13/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.015 BF
     113 route  112 110 108 107 	                   ->   13 x  13 x2048 
     114 conv    512       1 x 1/ 1     13 x  13 x2048 ->   13 x  13 x 512 0.354 BF
     115 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     116 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     117 conv    256       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 256 0.044 BF
     118 upsample                 2x    13 x  13 x 256 ->   26 x  26 x 256
     119 route  85 		                           ->   26 x  26 x 512 
     120 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     121 route  120 118 	                           ->   26 x  26 x 512 
     122 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     123 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     124 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     125 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     126 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     127 conv    128       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 128 0.044 BF
     128 upsample                 2x    26 x  26 x 128 ->   52 x  52 x 128
     129 route  54 		                           ->   52 x  52 x 256 
     130 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
     131 route  130 128 	                           ->   52 x  52 x 256 
     132 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
     133 conv    256       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 256 1.595 BF
     134 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
     135 conv    256       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 256 1.595 BF
     136 conv    128       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x 128 0.177 BF
     137 conv    256       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 256 1.595 BF
     138 conv     18       1 x 1/ 1     52 x  52 x 256 ->   52 x  52 x  18 0.025 BF
     139 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.20
    nms_kind: greedynms (1), beta = 0.600000 
     140 route  136 		                           ->   52 x  52 x 128 
     141 conv    256       3 x 3/ 2     52 x  52 x 128 ->   26 x  26 x 256 0.399 BF
     142 route  141 126 	                           ->   26 x  26 x 512 
     143 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     144 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     145 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     146 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     147 conv    256       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x 256 0.177 BF
     148 conv    512       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 512 1.595 BF
     149 conv     18       1 x 1/ 1     26 x  26 x 512 ->   26 x  26 x  18 0.012 BF
     150 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.10
    nms_kind: greedynms (1), beta = 0.600000 
     151 route  147 		                           ->   26 x  26 x 256 
     152 conv    512       3 x 3/ 2     26 x  26 x 256 ->   13 x  13 x 512 0.399 BF
     153 route  152 116 	                           ->   13 x  13 x1024 
     154 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     155 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     156 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     157 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     158 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
     159 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
     160 conv     18       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x  18 0.006 BF
     161 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
    nms_kind: greedynms (1), beta = 0.600000 
    Total BFLOPS 59.563 
    avg_outputs = 489778 
     Allocate additional workspace_size = 12.46 MB 
    Loading weights from backup/custom.weights...
     seen 64, trained: 236 K-images (3 Kilo-batches_64) 
    Done! Loaded 162 layers from weights-file 
    
     calculation mAP (mean average precision)...
     Detection layer: 139 - type = 28 
     Detection layer: 150 - type = 28 
     Detection layer: 161 - type = 28 
    388
     detections_count = 805, unique_truth_count = 512  
    class_id = 0, name = license_plate, ap = 89.80%   	 (TP = 439, FP = 45) 
    
     for conf_thresh = 0.25, precision = 0.91, recall = 0.86, F1-score = 0.88 
     for conf_thresh = 0.25, TP = 439, FP = 45, FN = 73, average IoU = 74.06 % 
    
     IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
     mean average precision (mAP@0.50) = 0.898026, or 89.80 % 
    Total Detection Time: 31 Seconds
    
    Set -points flag:
     `-points 101` for MS COCO 
     `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data) 
     `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom dataset
    

*   Precision: 91 %
*   Average Precision: 89.80 %
*   Recall: 86 %
*   F1-score: 88 %
*   Average IoU: 74.06 %
*   mAP@0.5: 89.80 %
*   Confusion Matrix:
    *   TP = 439
    *   FP = 45
    *   FN = 73
    *   unique_truth_count (TP+FN) = 512
    *   detections_count = 805


# Challenges

## Dataset



I spent quiet a lot of time for finding a well annotated and large dataset for licence plates, but I didn't succeed.  
I checked all these:


*   [kaggle](https://www.kaggle.com/andrewmvd/car-plate-detection)
  *   This is a small dataset with 433 images, which I think is not enough for model to converge well.


*   https://platerecognizer.com/number-plate-datasets/
  *   waste of time (none of them are free)


*   http://www.zemris.fer.hr/projects/LicensePlates/english/results.shtml
  *   small

*   http://www.inf.ufrgs.br/~crjung/alpr-datasets/
*   https://github.com/detectRecog/CCPD





## Wrong path!

I spent a lot of time for finding a tensorflow/keras implementation for YOLO. But none of them worked properly.

And running darknet on windows wasn't so simple,
so I tried to run darknet on colab.

## GPU

Google colab limited my GPU usage.  
It got slower and slower and slower and I couldn't finish my training.

# Code


```python
!pwd
```

    /content
    


```python
%cd drive/MyDrive/darknet
```

    /content/drive/MyDrive/darknet
    


```python

```
