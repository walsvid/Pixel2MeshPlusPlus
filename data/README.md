# Data and Models

## Pre-trained Models

### Download link
Google Drive: [https://drive.google.com/drive/folders/1bLhqXNoBxHh5PTbjoyqMnMtBzHwflL-q?usp=sharing](https://drive.google.com/drive/folders/1bLhqXNoBxHh5PTbjoyqMnMtBzHwflL-q?usp=sharing)

Direct Link: [http://www.sdspeople.fudan.edu.cn/fuyanwei/download/Pixel2MeshPlusPlus/](http://www.sdspeople.fudan.edu.cn/fuyanwei/download/Pixel2MeshPlusPlus/)

### Usage
The downloaded pre-training model zip file includes two components of our model: coarse shape generation and multi-view deformation network.

Please extract the model to the `coarse_mvp2m` and `refine_p2mpp` folders respectively according to the corresponding names. The folder structure after unzip should be as follows.

```
results
├── coarse_mvp2m
│   └── models
└── refine_p2mpp
    └── models
```

----

## Dataset
We use ShapeNet as our training and testing data. 

### Iamges
For input images, we use rendering images from [Choy et. al.](https://github.com/chrischoy/3D-R2N2).

Download image datasets and place them in a folder:
```
mkdir ShapeNetImages
wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
```
Please modify `train/test_image_path` to your 3D model path in the configuration file in `cfg/` before training.

### Ground-truth model
For ground-truth model, we adopt the dataset provided by [Wang et.al.](https://github.com/nywang16/Pixel2Mesh).
Specifically, our pre-process approach is sampling point cloud with vertex normal from origin ShapeNet 3D models.

When using the provided data make sure to respect the shapenet [license](https://shapenet.org/terms).

Download ground-truth models and place them in a folder:
```
mkdir ShapeNetModels
wget http://www.sdspeople.fudan.edu.cn/fuyanwei/download/Pixel2MeshPlusPlus/p2mpp_models.tar.gz
tar xzvf p2mpp_models.tar.gz
```
We also provided Google Drive [link](https://drive.google.com/drive/folders/1bLhqXNoBxHh5PTbjoyqMnMtBzHwflL-q?usp=sharing) for ground truth models data.

The zip file has already split data into train/test set. Please modify `train/test_data_path` to your 3D model path in the configuration file in `cfg/` before training.

