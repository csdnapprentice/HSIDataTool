# HSIDataTool(高光谱图像数据处理工具)
这是一个用于高光谱图像数据处理完整流程的代码仓库。它提供了数据预处理、中间训练过程以及训练后绘图的代码。这些代码可以满足基本需求。如果您希望使用这些代码，请注明原作者。希望这些代码能对您有所帮助。
This is a code repository designed for the complete workflow of hyperspectral image data processing. It provides code for data preprocessing, intermediate training processes, and post-training plotting. These codes can meet basic requirements. If you wish to use them, please credit the original author. I hope this code will be helpful to you.
## 使用方法
### RawDataProcess类
**类名**：`User`  
这是一个用户类，用于管理用户的基本信息和行为，例如注册、登录、更新信息等。

#### 2. 安装或引入
```python
# 假设你的类定义在 user.py 文件中
from user import User
```

#### 3. 创建类的实例
**构造函数**：`__init__(self, username: str, email: str)`  
**参数说明**：  
- `username`：用户的用户名，字符串类型。
- `email`：用户的电子邮件，字符串类型。

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
