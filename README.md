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

**示例**：
```python
# 创建用户对象
user = User(username="Tianyi", email="tianyi@example.com")
```

#### 4. 类的方法

##### 方法 1：`register(self, password: str) -> bool`  
**功能**：注册用户。  
**参数**：  
- `password`：用户的密码，字符串类型。  
**返回值**：布尔值，表示注册是否成功。  

**示例**：
```python
success = user.register(password="securepassword123")
if success:
    print("用户注册成功")
```

##### 方法 2：`login(self, password: str) -> bool`  
**功能**：用户登录。  
**参数**：  
- `password`：用户的密码，字符串类型。  
**返回值**：布尔值，表示登录是否成功。

##### 方法 3：`update_email(self, new_email: str) -> None`  
**功能**：更新用户的电子邮件地址。  
**参数**：  
- `new_email`：新的电子邮件地址，字符串类型。  


#### 5. 常见问题和注意事项


#### 6. 完整示例
