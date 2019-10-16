# Object Detection
This is an implementation of SSD for object detection in Tensorflow. It contains complete code for preprocessing, postprocessing, training and test. Besides, this repository is easy-to-use and can be developed on Linux and Windows.  

[SSD : Liu, Wei, et al. "Ssd: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016.](https://arxiv.org/abs/1512.02325)

## Getting Started
### 1 Prerequisites  
* Python3  
* Tensorflow  
* Opencv-python  
* Pandas  

### 2 Define your class names  
Download  and unzip this repository.  
`cd ../SSD/label`  
Open the `label.txt` and revise its class names as yours.  

### 3 Prepare images  
Copy your images and annotation files to directories `../SSD/data/annotation/images` and `../SSD/data/annotation/images/xml` respectively, where the annotations should be obtained by [a graphical image annotation tool](https://github.com/tzutalin/labelImg) and  saved as XML files in PASCAL VOC format.  
`cd ../SSD/Code`  
run  
`python spilt.py`  
Then train and val images will be generated in  `../SSD/data/annotation/train` and  `/SSD/data/annotation/test` directories, respectively.  

### 4 Generate anchors (default boxes)    
`cd ../SSD/code`  
run  
`python anchor_generators.py`  
Anchors generated will be saved in the directory `../SSD/anchor/anchor.txt`.  

### 5 Train model using Tensorflow  
The model parameters, training parameters and eval parameters are all defined by `parameters.py`.  
`cd ../SSD/code`  
run  
`python train.py`  
The model will be saved in directory `../SSD/model/checkpoint`, and some detection results are saved in `../SSD/pic`. 
 
### 6 Visualize model using Tensorboard  
`cd ../SSD`  
run  
`tensorboard --logdir=model/`   
Open the URL in browser to visualize graph of the model, as follows:

![Image](https://github.com/xiaogangLi/tensorflow-MobilenetV1-SSD/blob/master/SSD/pic/graph3.jpg)

### Local Graph
![Image](https://github.com/xiaogangLi/tensorflow-MobilenetV1-SSD/blob/master/SSD/pic/graph2.jpg)


## Examples  
Belows are some successful detection examples in my dataset:   

![Image](https://github.com/xiaogangLi/tensorflow-MobilenetV1-SSD/blob/master/SSD/pic/example1.jpg)
![Image](https://github.com/xiaogangLi/tensorflow-MobilenetV1-SSD/blob/master/SSD/pic/example3.jpg)
![Image](https://github.com/xiaogangLi/tensorflow-MobilenetV1-SSD/blob/master/SSD/pic/example2.jpg)

