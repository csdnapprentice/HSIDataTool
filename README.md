# HSIDataTool(高光谱图像数据处理工具)
这是一个用于高光谱图像数据处理完整流程的代码仓库。它提供了数据预处理、中间训练过程以及训练后绘图的代码。这些代码可以满足基本需求。如果您希望使用这些代码，请注明原作者。希望这些代码能对您有所帮助。
This is a code repository designed for the complete workflow of hyperspectral image data processing. It provides code for data preprocessing, intermediate training processes, and post-training plotting. These codes can meet basic requirements. If you wish to use them, please credit the original author. I hope this code will be helpful to you.
## 使用方法
### RawDataProcess类
#### 1. 简介
**类名**：`RawDataProcess`  
这是一个原始数据处理类，用于对高光谱图像数据和LiDAR的原始数据进行处理，将高光谱图像数据转化为一个一个的数据块，并封装成数据集，以供使用。

#### 2. 安装或引入
```python
from DataProcess.RawDataProcess import RawDataProcess
```

#### 3. 创建类的实例
**构造函数**：`__init__(self, patch_size, file_path, hsi_name, lidar_name, ground_truth_name, train_propotion, validate_propotion, PCA, PCA_dim)`  
**参数说明**：  
- `patch_size`：表示要切成的数据块的空间尺寸的大小为patch_size*patch_size的大小，这是一个正整数值。
- `file_path`：表示原始数据的文件路径，如'./trento_data.mat'。
- `hsi_name`：.mat 文件的数据字典中的键，表示高光谱图像数据。例如，在包含三个键（'HSI'，'LiDAR'，'GT'）的 trento_data.mat 文件中，hsi_name 的值应设置为 'HSI'。
- `lidar_name`：.mat 文件的数据字典中的键，表示 LiDAR 数据。例如，在包含三个键（'HSI'，'LiDAR'，'GT'）的 trento_data.mat 文件中，lidar_name 的值应设置为 'LiDAR'。
- `ground_truth_name`：.mat 文件的数据字典中的键，表示地面真值图像的数据。例如，在包含三个键（'HSI'，'LiDAR'，'GT'）的 trento_data.mat 文件中，ground_truth_name 的值应设置为 'GT'。
- `train_propotion`：表示训练集中每一个类别的样本的数量，这是一个列表，其中每一个值都为正整数。
- `validate_propotion`：表示验证集中每一个类别的样本的数量，这是一个列表，其中每一个值都为正整数，注：测试集是通过train_propotion和validate_propotion以及原始数据中的样本数量计算出来的。
- `PCA`：表示高光谱图像原始数据是否进行PCA。
- `PCA_dim`：表示高光谱图像原始数据经过PCA之后的维度。
#### 4. 类的可类外调用的方法
##### 方法 1：`data_process(self)`  
**功能**：处理并返回高光谱图像数据、LiDAR数据和地面真值图像的数据集。   
**返回值**：  
- `train_dataset`：训练数据集。  
- `validate_dataset`：验证数据集。  
- `test_dataset`：测试数据集。  
- `no_class_dataset`：无真实标签的数据集。  
- `whole_graph`：包含整个数据图所有像素点的以其为中心的数据块。  
- `index`：数据索引。（我在使用的时候没有用到）  
- `class_num`：类别数量。  
- `hsi_channels`：处理后高光谱图像通道数。  
- `lidar_channels`：处理后LiDAR数据通道数。

### ResultProcess类
#### 1. 简介
**类名**：`ResultProcess`  
这是一个模型输出结果数据的处理类，用于根据模型输出数据的混淆矩阵计算OA、AA、KAPPA等衡量指标。

#### 2. 安装或引入
```python
from DataProcess.ResultProcess import ResultProcess
```

#### 3. 创建类的实例
**构造函数**：`__init__(self)`  
**参数说明**：  
无参构造函数
#### 4. 类的可类外调用的方法
##### 方法 1：`compute(self, confusion_matrix)`  
**功能**：处理混淆矩阵并返回各项指标。   
**参数说明**：  
- `confusion_matrix`：模型结果数据的混淆矩阵。  
**返回值**：  
- `formatted_each_class_acc`：每一个类别的精度，是一个列表。  
- `formatted_OA`：总体精度。  
- `formatted_AA`：平均精度。  
- `formatted_K`：卡帕系数。  
### ModelOperator类
#### 1. 简介
**类名**：`ResultProcess`  
用于训练模型等各种操作。

#### 2. 安装或引入
```python
from ModelOperator.ModelOperator import ModelOperator
```

#### 3. 创建类的实例
**构造函数**：`__init__(self)`  
**参数说明**：  
无参构造函数
#### 4. 类的可类外调用的方法
##### 方法 1：`train_my_model(self, model, train, optimizer, class_num, device, no_class, epoch)`  
**功能**：处理并返回高光谱图像数据、LiDAR数据和地面真值图像的数据集。   
**参数说明**：  
- `confusion_matrix`：模型结果数据的混淆矩阵。  
**返回值**：  
- `formatted_each_class_acc`：每一个类别的精度，是一个列表。  
- `formatted_OA`：总体精度。  
- `formatted_AA`：平均精度。  
- `formatted_K`：卡帕系数。
  ##### 方法 1：`compute(self, confusion_matrix)`  
**功能**：处理并返回高光谱图像数据、LiDAR数据和地面真值图像的数据集。   
**参数说明**：  
- `confusion_matrix`：模型结果数据的混淆矩阵。  
**返回值**：  
- `formatted_each_class_acc`：每一个类别的精度，是一个列表。  
- `formatted_OA`：总体精度。  
- `formatted_AA`：平均精度。  
- `formatted_K`：卡帕系数。  
