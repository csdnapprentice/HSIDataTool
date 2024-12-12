import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA  # 进行数据的降维
import torch.utils.data as data
from torch.utils.data import Dataset
import torch
import random

class MultiModalDataset(Dataset):
    def __init__(self, HSI, DSM, LABEL):
        self.HSI = HSI  # HSI数据
        self.DSM = DSM  # DSM数据
        self.LABEL = LABEL  # 标签

    def __len__(self):
        # 返回数据集的长度（样本数量）
        return len(self.HSI)

    def __getitem__(self, index):
        # 获取索引为index的样本
        hsi = self.HSI[index]  # 获取HSI数据
        dsm = self.DSM[index]  # 获取DSM数据
        label = self.LABEL[index]  # 获取标签数据

        # 返回多模态样本
        return hsi, dsm, label

class RawDataProcess:
    def __init__(self, patch_size, file_path, hsi_name, lidar_name, ground_truth_name, train_propotion, validate_propotion, PCA, PCA_dim):
        self.patch_size = patch_size
        self.file_path = file_path
        self.hsi_name = hsi_name
        self.lidar_name = lidar_name
        self.ground_truth_name = ground_truth_name
        self.train_propotion = train_propotion
        self.validate_propotion = validate_propotion
        self.pca = PCA
        self.PCA_dim = PCA_dim
    def load_data(self, file_path, hsi_name, lidar_name, ground_truth_name):
        HSI = sio.loadmat(file_path)[hsi_name].astype("float32")
        LiDAR = sio.loadmat(file_path)[lidar_name].astype("float32")
        ground_truth = sio.loadmat(file_path)[ground_truth_name].astype("uint8")
        return HSI, LiDAR, ground_truth

    def DataNormalization(self, hsi, lidar):
        hsi = hsi.transpose(2,0,1)
        lidar = lidar.transpose(2,0,1)
        # 按照光谱维度进行归一化，保留光谱维度之间的差异性
        for i in range(hsi.shape[0]):
            max = np.max(hsi[i])
            min = np.min(hsi[i])
            hsi[i] = (hsi[i] - min) / (max - min)
        for i in range(lidar.shape[0]):
            max = np.max(lidar[i])
            min = np.min(lidar[i])
            lidar[i] = (lidar[i] - min) / (max - min)
        ## 整个数据进行归一化
        # max = np.max(hsi)
        # min = np.min(hsi)
        # hsi = (hsi - min) / (max - min)
        # max = np.max(lidar)
        # min = np.min(lidar)
        # lidar = (lidar - min) / (max - min)
        lidar = lidar.transpose(1,2,0)
        hsi = hsi.transpose(1,2,0)
        return hsi, lidar

    def data_PCA(self, input_data, dim):
        newdata = input_data.reshape(-1, input_data.shape[2])
        pca = PCA(n_components=dim, whiten=True)
        newdata = pca.fit_transform(newdata)
        newdata = newdata.reshape(input_data.shape[0], input_data.shape[1], dim)
        return newdata

    def cut_patch(self, HSI, LiDAR, ground_truth, patch_size):
        # 获取整张图中的类别数
        class_num = np.max(ground_truth)
        # 获取原始数据图片的形状
        m, n, z1 = HSI.shape
        _,_,lidar_channel = LiDAR.shape
        # 找到所有的非零值
        index = np.where(ground_truth)
        # 对高光谱图像周围做填充
        HSI = torch.from_numpy(np.pad(HSI, pad_width=patch_size//2))
        # 对LiDAR数据周围做填充
        LiDAR = torch.from_numpy(np.pad(LiDAR, pad_width=patch_size//2))
        # 对于 gt 图周围做填充
        ground_truth = torch.from_numpy(np.pad(ground_truth, pad_width=patch_size//2))
        # 创建数组保存小块数据
        class_patch = []
        align = 1
        half_patch_size = patch_size//2
        # 创建列表用于创建整张图
        whole_graph = [[], [], []]
        no_class_data = [[],[],[]]
        if patch_size//2 == 0:
            align = 0
        for i in range(half_patch_size, m+half_patch_size):
            for j in range(half_patch_size, n+half_patch_size):
                whole_graph[0].append(HSI[i-half_patch_size:i+half_patch_size+align,
                                      j-half_patch_size:j+half_patch_size+align,
                                      half_patch_size:z1+half_patch_size].permute(2, 0, 1))
                whole_graph[1].append(LiDAR[i-half_patch_size:i+half_patch_size+align,
                                      j - half_patch_size:j + half_patch_size + align,
                                      half_patch_size:lidar_channel+half_patch_size].permute(2, 0, 1))
                whole_graph[2].append(0)
                if ground_truth[i][j] > 0:
                    while ground_truth[i][j] > len(class_patch):
                        class_patch.append([[], [], []])
                    class_patch[ground_truth[i][j]-1][0].append(HSI[i-half_patch_size:i+half_patch_size+align,
                                                                j-half_patch_size:j+half_patch_size+align,
                                                                half_patch_size:z1+half_patch_size].permute(2, 0, 1))
                    class_patch[ground_truth[i][j]-1][1].append(LiDAR[i-half_patch_size:i+half_patch_size+align,
                                                                j - half_patch_size:j + half_patch_size + align,
                                                                half_patch_size:lidar_channel+half_patch_size].permute(2, 0, 1))
                    class_patch[ground_truth[i][j]-1][2].append(np.eye(class_num)[ground_truth[i][j]-1])
                    # 标签都减去1，得到将标签转换成的新标签
                elif ground_truth[i][j] == 0:
                    no_class_data[0].append(HSI[i-half_patch_size:i+half_patch_size+align,
                                                                j-half_patch_size:j+half_patch_size+align,
                                                                half_patch_size:z1+half_patch_size].permute(2, 0, 1))
                    no_class_data[1].append(LiDAR[i-half_patch_size:i+half_patch_size+align,
                                                                j - half_patch_size:j + half_patch_size + align,
                                                                half_patch_size:lidar_channel+half_patch_size].permute(2, 0, 1))
                    no_class_data[2].append(np.zeros(class_num))
        return class_patch, whole_graph, no_class_data, index


    def data_fusion(self, dataloader_list):
        # 将hsi、lidar、gt等数据封装在一块
        hsi = []
        lidar = []
        label = []
        for i in range(len(dataloader_list)):
            for fuse_data in dataloader_list[i]:
                hsi.append(fuse_data[0])
                lidar.append(fuse_data[1])
                label.append(fuse_data[2])
        dataset = MultiModalDataset(hsi, lidar, label)
        return dataset


    def data_split(self, dataset, train_propotion, validate_propotion, no_class_data):
        class_num = len(dataset)
        train_class = []  # 用来保存随机抽取的训练数据
        validate_class = []  # 用来保存随机抽取的验证数据
        test_class = []  # 用来保存随机抽取出的测试数据
        have_class_data_num = 0
        for i in range(len(dataset)):
            single_dataset = MultiModalDataset(dataset[i][0], dataset[i][1], dataset[i][2])  # 合成，直接加载，这样就能直接得到这些数据
            single_train, single_validate, single_test = data.random_split(single_dataset,
                                                                           [train_propotion[i], validate_propotion[i],
                                                                            len(single_dataset)
                                                                            - train_propotion[i]
                                                                            - validate_propotion[i]])
            train_class.append(single_train)
            validate_class.append(single_validate)
            test_class.append(single_test)
            have_class_data_num += single_validate.__len__()
            have_class_data_num += single_train.__len__()
            have_class_data_num += single_test.__len__()
        no_class_data = MultiModalDataset(no_class_data[0], no_class_data[1], no_class_data[2])

        train_data = self.data_fusion(train_class)
        validate_data = self.data_fusion(validate_class)
        test_data = self.data_fusion(test_class)

        # 将hsi、lidar、gt等数据封装在一块
        hsi = []
        lidar = []
        label = []
        for fuse_data in train_data:
            hsi.append(fuse_data[0])
            lidar.append(fuse_data[1])
            label.append(fuse_data[2])
        train_data = MultiModalDataset(hsi, lidar, label)
        hsi = []
        lidar = []
        label = []
        random_number = random.sample(range(len(no_class_data)), len(train_data) * 5)
        for i in random_number:
            hsi.append(no_class_data[i][0])
            lidar.append(no_class_data[i])
            label.append(no_class_data[i])
        no_class_data = MultiModalDataset(hsi, lidar, label)
        print(train_data.__len__())
        return train_data, validate_data, test_data,no_class_data, class_num


    def data_process(self):
        print("开始处理数据")
        # 数据加载
        HSI, LiDAR, ground_truth = self.load_data(self.file_path, self.hsi_name, self.lidar_name, self.ground_truth_name)
        if len(LiDAR.shape) == 2:
            LiDAR = LiDAR.reshape(LiDAR.shape[0], LiDAR.shape[1], 1)
        lidar_channels = LiDAR.shape[2]
        HSI, LiDAR = self.DataNormalization(HSI, LiDAR)
        if self.pca:
            HSI = self.data_PCA(HSI, self.PCA_dim)
            LiDAR = self.data_PCA(LiDAR, lidar_channels)
        hsi_channels = HSI.shape[2]
        class_data, whole_graph, no_class_data,index = self.cut_patch(HSI, LiDAR, ground_truth, self.patch_size)
        train_dataset, validate_dataset, test_dataset,no_class_dataset, class_num = self.data_split(class_data, self.train_propotion, self.validate_propotion, no_class_data)
        print("类别数为：", class_num)
        return train_dataset, validate_dataset, test_dataset,no_class_dataset, whole_graph, index, class_num, hsi_channels, lidar_channels