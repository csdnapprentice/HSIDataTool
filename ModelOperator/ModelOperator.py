import torch
import numpy as np

class ModelOperator:
    def train_my_model(self, model, train, optimizer, class_num, device, no_class, epoch):
        train_acc = 0
        all_loss_train = 0
        confusion_matrix = np.zeros((class_num, class_num))
        if epoch < 100:
            for data in train:
                hsi, lidar, label = data
                hsi, lidar, label = hsi.to(device), lidar.to(device), label.to(device).argmax(1)
                output = model(hsi, lidar)
                loss = model.loss_function(output, label)
                all_loss_train += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                for i in range(output["result"].shape[0]):
                    confusion_matrix[label[i]][output["result"][i].argmax(0)] += 1
                train_acc += (output["result"].argmax(1) == label).sum().item()
            train_acc = train_acc / len(train.dataset)
            all_loss_train = all_loss_train / len(train)
        else:
            for data in train:
                hsi, lidar, label = data
                hsi, lidar, label = hsi.to(device), lidar.to(device), label.to(device).argmax(1)
                output = model(hsi, lidar)
                loss = model.loss_function(output, label)
                all_loss_train += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                for i in range(output["result"].shape[0]):
                    confusion_matrix[label[i]][output["result"][i].argmax(0)] += 1
                train_acc += (output["result"].argmax(1) == label).sum().item()
            train_acc = train_acc / len(train.dataset)
            all_loss_train = all_loss_train / len(train)

        return train_acc, all_loss_train, confusion_matrix


    def validate_my_model(self, model, validate, class_num, validate_acc_list, count,device):
        all_loss_validate = 0
        validate_acc = 0
        confusion_matrix = np.zeros((class_num, class_num))
        with torch.no_grad():
            for data in validate:
                hsi, lidar, label = data
                hsi, lidar, label = hsi.to(device), lidar.to(device), label.to(device).argmax(1)
                output = model(hsi, lidar)
                loss = model.loss_function(output, label) + model.loss_function2(output, label)
                for i in range(output["result"].shape[0]):
                    confusion_matrix[label[i]][output["result"][i].argmax(0)] += 1
                all_loss_validate += loss.item()
                validate_acc += (output[0].argmax(1) == label).sum().item()
        validate_acc = validate_acc / len(validate.dataset)
        all_loss_validate = all_loss_validate / len(validate)
        if validate_acc_list:
            if validate_acc > max(validate_acc_list):
                torch.save(model.state_dict(), 'model_trento.pth')
                count = 0
            if validate_acc < max(validate_acc_list):
                count += 1
        else:
            torch.save(model.state_dict(), 'model_trento.pth')
        return validate_acc, all_loss_validate, confusion_matrix, count


    def test_my_model(self, model, test, class_num,device):
        test_acc = 0
        confusion_matrix = np.zeros((class_num, class_num))
        with torch.no_grad():
            for data in test:
                hsi, lidar, label = data
                hsi, lidar, label = hsi.to(device), lidar.to(device), label.to(device).argmax(1)
                output = model(hsi, lidar)
                for i in range(output["result"].shape[0]):
                    confusion_matrix[label[i]][output["result"][i].argmax(0)] += 1
                test_acc += (output["result"].argmax(1) == label).sum().item()
        test_acc = test_acc/len(test.dataset)
        return test_acc, confusion_matrix
