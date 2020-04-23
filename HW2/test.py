import numpy as np
# from keras.models import Sequential
# from keras.layers import Conv2D
import torch
from torch import nn
import csv
# from PIL import Image
import pandas as pd, numpy as np, matplotlib, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from skimage import io
from skimage import color
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
import time
import copy
import pdb
# import torch.nn.functional as F
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import gc; gc.collect() 

# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def load_data_and_get_class(path_to_data):
    data = pd.read_csv(path_to_data)
    encoder = LabelEncoder()
    data['Class'] = encoder.fit_transform(data['Finding Labels'])
    return data

class ChestXrayDataset(Dataset):
    """Chest X-ray dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = load_data_and_get_class(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])

        image = io.imread(img_name,as_gray=True)
        
        image = (image - image.mean()) / image.std()
            
        image_class = self.data_frame.iloc[idx, -1]

        sample = {'x': image[None,:], 'y': image_class}

        if self.transform:
            sample = self.transform(sample)

        sample

        return sample





train_df_path = './HW2_trainSet_new.csv'
val_df_path = './HW2_validationSet_new.csv'
test_df_path = './HW2_testSet_new.csv'
root_dir = '/beegfs/ga4493/data/HW2/images/'
# test_df_path = './test.csv'

transformed_dataset = {'train': ChestXrayDataset(csv_file = train_df_path, root_dir = root_dir),
                       'validate': ChestXrayDataset(csv_file = val_df_path, root_dir = root_dir),
                       'test': ChestXrayDataset(csv_file = val_df_path, root_dir = root_dir)}
                                          
bs = 9
dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs,
                        shuffle=True, drop_last=True, pin_memory=False) for x in ['train', 'validate'
#                                                                ,'test'
                                                              ]}
data_sizes ={x: len(transformed_dataset[x]) for x in ['train', 'validate'
#                                                       ,'test'
                                                     ]}

def train_model(model, dataloader, optimizer, scheduler, loss_fn, num_epochs = 50, verbose = False):
    acc_dict = {'train':[],'validate':[]}
    loss_dict = {'train':[],'validate':[]}
    best_acc = 0
    phases = ['train','validate']
    since = time.time()
    for i in range(num_epochs):
        print('Epoch: {}/{}'.format(i, num_epochs-1))
        print('-'*10)
        for p in phases:
            running_correct = 0
            running_loss = 0
            running_total = 0
            if p == 'train':
                model.train()
            else:
                model.eval()
                
            for data in dataloader[p]:
                #explore if no grad will get me to run
                optimizer.zero_grad()
                image = data['x'].cuda()
                label = data['y'].cuda()
                # with torch.set_grad_enabled(p == 'train'):
                #     pdb.set_trace()
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                loss = loss_fn(outputs, label)
                # output = model(image)
                # loss = loss_fn(output, label)
                # _, preds = torch.max(output, dim = 1)
                num_imgs = image.size()[0]
                running_correct += torch.sum(preds ==label).item()
                running_loss += loss.item()*num_imgs
                running_total += num_imgs
                if p== 'train':
                    loss.backward()
                    optimizer.step()
            epoch_acc = float(running_correct/running_total)
            epoch_loss = float(running_loss/running_total)
            if verbose or (i%10 == 0):
                print('Phase:{}, epoch loss: {:.4f} Acc: {:.4f}'.format(p, epoch_loss, epoch_acc))

            # deep copy the model            
            acc_dict[p].append(epoch_acc)
            loss_dict[p].append(epoch_loss)
            if p == 'validate':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'model_dict')
            else:
                if scheduler:
                    scheduler.step()
        #checkpoint
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, 'checkpoint_dict')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))
    

    model.load_state_dict(best_model_wts)
    
    return model, acc_dict, loss_dict


class Conv_model(nn.Module):
    def __init__(self):
        super(Conv_model,self).__init__()
        self.conv_layer1 =  nn.Sequential(
                            conv3x3(1, 16, stride=2),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2))
        self.resnet1 = BasicBlock(16, 16, stride=1)
        
        self.conv_layer2 =  nn.Sequential(
                            conv3x3(16, 32, stride = 2),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2))
        self.resnet2 = BasicBlock(32, 32, stride=1)
        
        self.conv_layer3 =  nn.Sequential(
                            conv3x3(32, 64, stride = 2),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2))
        self.resnet3 = BasicBlock(64, 64, stride=1)
        
        self.avgpooling = nn.MaxPool2d(kernel_size=16)
        
        self.fc2 = nn.Linear(64, 3)
        
    def forward(self,x):
        x = self.conv_layer1(x)
        # print(f'feature map after convolution operation with shape: {x.shape}')
        x = self.resnet1(x)
        # print(f'feature map after resnet1 operation with shape: {x.shape}')
        x = self.conv_layer2(x)
        # print(f'feature map after convolution operation with shape: {x.shape}')
        x = self.resnet2(x)
        # print(f'feature map after resnet2 operation with shape: {x.shape}')
        x = self.conv_layer3(x)
        # print(f'feature map after convolution operation with shape: {x.shape}')
        x = self.resnet3(x)
        # print(f'feature map after resnet3 operation with shape: {x.shape}')
#         x = out.reshape(out.size(0), -1)
        x = self.avgpooling(x)
        # print(f'feature map after avgpooling operation with shape: {x.shape}')
        # https://discuss.pytorch.org/t/how-to-resolve-runtime-error-due-to-tensor-size-mismatch/15815
        # x = x.view(x.size(0), -1)
        x = self.fc2(x.view(-1, 64))
        # print(f'feature map after linear operation with shape: {x.shape}')
        return x

model = Conv_model().double()
model.cuda()
cel = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.01)
lambda_func = lambda epoch: 0.5 ** epoch
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

try:
    checkpoint = torch.load('/checkpoint_dict')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
except:
    print('no checkpoint yet available')


model, acc_dict, loss_dict = train_model(model, dataloader, optimizer, scheduler, cel, num_epochs=10, verbose = True)

w = csv.writer(open("loss_dict.csv", "w"))
for key, val in loss_dict.items():
    w.writerow([key, val])

w = csv.writer(open("acc_dict.csv", "w"))
for key, val in acc_dict.items():
    w.writerow([key, val])

pd.DataFrame(loss_dict).plot()
plt.xlabel('epoch')
plt.ylim(bottom=0)
plt.title('loss')
plt.savefig('loss.png')

pd.DataFrame(acc_dict).plot()
plt.xlabel('epoch')
plt.ylim(bottom=0)
plt.title('acc')
plt.savefig('acc.png')


def evaluate_model(model, dataloader,loss_fn, phase = 'test', acc_and_loss=True):
    model.eval()
    actuals = []
    predictions = []
    running_correct = 0
    running_loss = 0
    running_total = 0
    for data in dataloader[phase]:
        image = data['x'].cuda()
        label = data['y'].cuda()
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        loss = loss_fn(outputs, label)
        num_imgs = image.size()[0]
        running_correct += torch.sum(preds ==label).item()
        running_loss += loss.item()*num_imgs
        running_total += num_imgs
        actuals.extend(target.view_as(prediction))
        predictions.extend(prediction)
    accuracy = float(running_correct/running_total)
    loss = float(running_loss/running_total)
    if acc_and_loss:
        return accuracy, loss 
    else:
        return [i.item() for i in actuals], [i.item() for i in predictions]


acc, loss = evaluate_model(model, dataloader,cel,phase = 'test')

print(f'accuracy as a result of test dataset on model {acc}')
print(f'loss as a result of test dataset on model {loss}')


# http://bytepawn.com/solving-mnist-with-pytorch-and-skl.html

actuals, predictions = evaluate_model(model, dataloader,cel,phase = 'test', acc_and_loss=False)
print('Confusion matrix:')
print(confusion_matrix(actuals, predictions))
print('F1 score: %f' % f1_score(actuals, predictions, average='micro'))
print('Accuracy score: %f' % accuracy_score(actuals, predictions))


def test_class_probabilities(model, device, test_loader, which_class):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data in dataloader[phase]:
            image = data['x'].cuda()
            label = data['y'].cuda()
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            actuals.extend(target.view_as(preds) == which_class)
            probabilities.extend(np.exp(output[:, which_class]))
    return [i.item() for i in actuals], [i.item() for i in probabilities]

for class_label in [0,1,2]:s
    actuals, class_probabilities = test_class_probabilities(model, device, test_loader, class_label)

fpr, tpr, _ = roc_curve(actuals, class_probabilities)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for digit=%d class' % which_class)
plt.legend(loc="lower right")
plt.show()
plt.savefig('roc_plot.png')

