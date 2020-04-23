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
from sklearn.metrics import confusion_matrix, roc_curve, f1_score, auc, accuracy_score
# from sklearn.model_selection import train_test_split
import time
import copy
import pdb
from torchvision import transforms, models
# import torch.nn.functional as F
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import gc; gc.collect() 

model_name = 'resnetfull'


# train_label_batch = torch.from_numpy(train_label_batch)
# train_label_batch = train_label_batch.type(torch.FloatTensor) 
# train_label_batch = train_label_batch.cuda()


# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def load_data_and_get_class(path_to_data):
    data = pd.read_csv(path_to_data)
    encoder = LabelEncoder()
    data['Class'] = encoder.fit_transform(data['Finding Labels'])
    return data

train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(896),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(896),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class ChestXrayDataset_ResNet(Dataset):
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
        
        image = io.imread(img_name)
        if len(image.shape) > 2 and image.shape[2] == 4:
            image = image[:,:,0]
            
        image=np.repeat(image[None,...],3,axis=0)

        image_class = self.data_frame.iloc[idx, -1]

        if self.transform:
            image = self.transform(image)
            
        sample = {'x': image, 'y': image_class}

        return sample


train_df_path = './HW2_trainSet_new.csv'
val_df_path = './HW2_validationSet_new.csv'
test_df_path = './HW2_testSet_new.csv'
root_dir = '/beegfs/ga4493/data/HW2/images/'

transformed_dataset = {'train': ChestXrayDataset_ResNet(csv_file = train_df_path, root_dir = root_dir),
                       'validate': ChestXrayDataset_ResNet(csv_file = val_df_path, root_dir = root_dir),
                       'test': ChestXrayDataset_ResNet(csv_file = val_df_path, root_dir = root_dir)}
                                          
bs = 9
dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs,
                        shuffle=True, drop_last=True, pin_memory=False) for x in ['train', 'validate'
                                                               ,'test'
                                                              ]}
data_sizes ={x: len(transformed_dataset[x]) for x in ['train', 'validate'
                                                      ,'test'
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
                image = data['x'].to(device=device, dtype=torch.float).cuda()
                label = data['y'].cuda()
                # with torch.set_grad_enabled(p == 'train'):
                #     pdb.set_trace()
                print(type(image))
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
                    torch.save(model.state_dict(), f'model_dict_{model_name}')
            else:
                if scheduler:
                    scheduler.step()
        #checkpoint
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, f'checkpoint_dict_{model_name}')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))
    

    model.load_state_dict(best_model_wts)
    
    return model, acc_dict, loss_dict


model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=False)
model.cuda()
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 3)

model = model.to(device)

cel = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr = 0.01)
lambda_func = lambda epoch: 0.5 ** epoch
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

# try:
#     checkpoint = torch.load('/checkpoint_dict')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']
# except:
#     print('no checkpoint yet available')


model, acc_dict, loss_dict = train_model(model, dataloader, optimizer, scheduler, cel, num_epochs=20, verbose = True)

w = csv.writer(open(f"loss_dict_{model_name}.csv", "w"))
for key, val in loss_dict.items():
    w.writerow([key, val])

w = csv.writer(open("acc_dict.csv", "w"))
for key, val in acc_dict.items():
    w.writerow([key, val])

pd.DataFrame(loss_dict).plot()
plt.xlabel('epoch')
plt.ylim(bottom=0)
plt.title('loss')
plt.savefig(f'loss_{model_name}.png')

pd.DataFrame(acc_dict).plot()
plt.xlabel('epoch')
plt.ylim(bottom=0)
plt.title('acc')
plt.savefig(f'acc_{model_name}.png')


def evaluate_model(model, dataloader,loss_fn, phase = 'test', acc_and_loss=True):
    model.eval()
    actuals = []
    predictions = []
    running_correct = 0
    running_loss = 0
    running_total = 0
    for data in dataloader[phase]:
        image = data['x'].to(device=device, dtype=torch.float).cuda()
        label = data['y'].cuda()
        outputs = model(image.to(device=device, dtype=torch.float))
        _, preds = torch.max(outputs, 1)
        loss = loss_fn(outputs, label)
        num_imgs = image.size()[0]
        running_correct += torch.sum(preds ==label).item()
        running_loss += loss.item()*num_imgs
        running_total += num_imgs
        actuals.extend(label.view_as(preds))
        predictions.extend(preds)
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


def test_class_probabilities(model, dataloader, which_class, phase = 'test'):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data in dataloader[phase]:
            image = data['x'].to(device=device, dtype=torch.float).cuda()
            label = data['y'].cuda()
            outputs = model(image.to(device=device, dtype=torch.float))
            _, preds = torch.max(outputs, 1)
            actuals.extend(label.view_as(preds) == which_class)
            probabilities.extend(np.exp(outputs[:, which_class].cpu()))
    return [i.item() for i in actuals], [i.item() for i in probabilities]

for class_label in [0, 1, 2]:
    actuals, class_probabilities = test_class_probabilities(model, dataloader, class_label)

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
    plt.title('ROC for digit=%d class' % class_label)
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(f'roc_plot{model_name}.png')

