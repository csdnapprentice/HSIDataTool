import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from sklearn.manifold import TSNE
from datetime import datetime
class Paint:
    def __init__(self):
        self.trento_color =  [[0, 217, 89], [203, 26, 0], [251, 118, 19], [51, 154, 26], [51, 152, 26], [0, 0, 255]]
        self.houston_color =  [[84,172,72], [69,93,33], [56,111,77], [60,132,70], [146,82,53], [104,189,200], [255,255,255],
              [200,177,202], [219,50,43], [121,36,37], [55,101,166], [255,220,84], [219,143,52], [85,48,126], [228,119,91]]
        self.dataset = []
    def paint_classfication_graph(self,whichGraph,index, whole_graph, model, file_name):
        if whichGraph == 'Trento':
            self.paint_trento_graph(index, whole_graph, model, file_name)
        elif whichGraph == 'Houston':
            self.paint_houston_graph(index, whole_graph, model, file_name)
    def paint_trento_graph(self, index, whole_graph, model, file_name):
        color = self.trento_color
        color = torch.tensor(color, dtype=torch.float32)
        color = color / 255
        color = color.tolist()
        paint_graph = np.zeros((166, 600, 3))
        length, width, spec = paint_graph.shape
        for i in range(length):
            hsi = torch.stack(whole_graph[0][i*width:(i+1) * width]).to("cuda:0")
            lidar = torch.stack(whole_graph[1][i*width:(i+1) * width]).to("cuda:0")
            label = torch.tensor(whole_graph[2][i*width:(i+1) * width]).to("cuda:0")
            output= model(hsi, lidar)
            output = output["result"].argmax(1).tolist()
            paint_graph[i][:] = [color[i]  for i in output]
        if not os.path.exists("./fig/classification"):  # 如果不存在此路径，则创建此文件夹路径
            os.makedirs("./fig/classification")
        plt.imshow(paint_graph)
        plt.axis('off')
        plt.savefig("./fig/classification/" + file_name, dpi=600, bbox_inches='tight', pad_inches=0)
        plt.show()
    def paint_houston_graph(self, index, whole_graph, model, file_name):
        color = self.houston_color
        color = torch.tensor(color, dtype=torch.float32)
        color = color / 255
        color = color.tolist()
        paint_graph = np.zeros((349, 1905, 3))
        length, width, spec = paint_graph.shape
        for i in range(length):
            hsi = torch.stack(whole_graph[0][i*width:(i+1) * width]).to("cuda")
            lidar = torch.stack(whole_graph[1][i*width:(i+1) * width]).to("cuda")
            label = torch.tensor(whole_graph[2][i*width:(i+1) * width]).to("cuda")
            output = model(hsi, lidar)
            output = output["result"].argmax(1).tolist()
            paint_graph[i][:] = [color[i]  for i in output]
        if not os.path.exists("./fig/classification"):  # 如果不存在此路径，则创建此文件夹路径
            os.makedirs("./fig/classification")
        plt.imshow(paint_graph)
        plt.axis('off')
        plt.savefig("./fig/classification/" + file_name, dpi=600, bbox_inches='tight', pad_inches=0)
        plt.show()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




