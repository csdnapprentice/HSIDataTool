import torch
import matplotlib.pyplot as plt
import os


class ResultProcess:
    def compute(self, confusion_matrix):
        each_class_acc = []
        acc = 0
        all = 0
        for i in range(len(confusion_matrix)):
            each_class_acc.append(confusion_matrix[i][i]/sum(confusion_matrix[i]) * 100)  # 每一个类别的精度
            acc += confusion_matrix[i][i]
            all += sum(confusion_matrix[i])
        # 接下来计算总体精度
        OA = acc/all * 100
        # 再计算平均精度
        AA = sum(each_class_acc)/len(each_class_acc)
        # 计算卡帕系数
        each_colum = []
        for i in range(len(confusion_matrix)):
            each_colum.append(0)
            for j in range(len(confusion_matrix)):
                each_colum[i] += confusion_matrix[j][i]
            each_colum[i] = each_colum[i]*sum(confusion_matrix[i])
        Pe = sum(each_colum)/(all**2)
        K = (OA/100-Pe)/(1-Pe)
        K = K * 100
        formatted_each_class_acc = [f"{acc:.3f}" for acc in each_class_acc]
        formatted_OA = f"{OA:.3f}"
        formatted_AA = f"{AA:.3f}"
        formatted_K = f"{K:.3f}"
        return formatted_each_class_acc, formatted_OA, formatted_AA, formatted_K

    def paint_classification_graph(self, index, whole_graph, model, file_name, color, paint_graph, device):
        color = torch.tensor(color, dtype=torch.float32)
        color = color/255
        color = color.tolist()
        num = 0
        print(len(index[0]))
        for i in range(len(index[0])):
            hsi = torch.tensor(whole_graph[0][index[0][i]*600 + index[1][i]]).to(device)
            hsi = hsi.reshape(1, hsi.shape[0], hsi.shape[1], hsi.shape[2])
            lidar = torch.tensor(whole_graph[1][index[0][i]*600 + index[1][i]]).to(device)
            lidar = lidar.reshape(1, 1, lidar.shape[1], lidar.shape[1])
            label = torch.tensor(whole_graph[2][index[0][i]*600 + index[1][i]]).to(device)
            output = model(hsi, lidar)
            paint_graph[index[0][num]][index[1][num]] = color[output.argmax(1).item()]
            num += 1
        if not os.path.exists("./fig/"):  # 如果不存在此路径，则创建此文件夹路径
            os.mkdir("./fig/")
        plt.savefig("./fig/"+file_name, dpi=600)
        plt.imshow(paint_graph)
        plt.axis('off')
        plt.show()