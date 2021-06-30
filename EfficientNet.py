# Install EFFICIENTNET
!pip install efficientnet_pytorch


# Load TRAIN, VALIDATION and TEST data sets
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image
import torchvision
import numpy
import seaborn
import pandas


class LabeledDataset(Dataset):
    def __init__(self, labels_file, data_dir):
        self.filenames, self.labels = [], []
        self.data_dir = data_dir
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])
        file = open(labels_file,'r')
        for line in file:
            filename, label = line.split(',')
            label = label[0]
            self.filenames.append(filename)
            self.labels.append(label)
        file.close()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = cv2.imread(self.data_dir + '/' + self.filenames[idx])
        image = self.transform(image)
        label = self.labels[idx]
        label = torch.from_numpy(numpy.array(int(label)))
        return (image, label)


class UnlabeledDataset(Dataset):
    def __init__(self, labels_file, data_dir):
        self.filenames, self.labels = [], []
        self.data_dir = data_dir
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])
        file = open(labels_file,'r')
        for line in file:
            filename = line.split()[0]
            self.filenames.append(filename)
        file.close()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = cv2.imread(self.data_dir + '/' + self.filenames[idx])
        image = self.transform(image)
        return image
    
    
train_dataset = LabeledDataset('../input/ai-unibuc-23-31-2021/train.txt', '../input/ai-unibuc-23-31-2021/train')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

validation_dataset = LabeledDataset('../input/ai-unibuc-23-31-2021/validation.txt', '../input/ai-unibuc-23-31-2021/validation')
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=32, shuffle=True)

test_dataset = UnlabeledDataset('../input/ai-unibuc-23-31-2021/test.txt', '../input/ai-unibuc-23-31-2021/test')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32)


# Load PRETRAINED EFFICIENTNET and TRAIN WITH CUSTOM DATA
from efficientnet_pytorch import EfficientNet
import torch 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_epoch_validation_loss = 100.0


for epoch in range(1, 10):
    epoch_train_loss = 0.0    
    model.train()
    for batch_idx, (images_batch, labels_batch) in enumerate(train_dataloader):
        optimizer.zero_grad()
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
        labels_predictions = model(images_batch)
        loss = criterion(labels_predictions, labels_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        
    epoch_validation_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (images_batch, labels_batch) in enumerate(validation_dataloader):
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
            labels_predictions = model(images_batch)
            loss = criterion(labels_predictions, labels_batch)
            epoch_validation_loss += loss.item()

    epoch_train_loss = epoch_train_loss / len(train_dataloader.dataset)
    epoch_validation_loss = epoch_validation_loss / len(validation_dataloader.dataset)
    print(f'Epoch: {epoch} Train Loss: {epoch_train_loss} Validation Loss: {epoch_validation_loss}')
        
    if epoch_validation_loss <= best_epoch_validation_loss:
        checkpoint = {
            'state_dict': model.state_dict()
        }
        torch.save(checkpoint, './checkpoint.pt')
        print(f'Saved checkpoint. Validation loss decreased from {best_epoch_validation_loss} to {epoch_validation_loss}')
        best_epoch_validation_loss = epoch_validation_loss


# Find ACCURACY on VALIDATION and create CONFUSION MATRIX
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
import matplotlib.pyplot as plt


checkpoint = torch.load('./checkpoint.pt')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
smax = nn.Softmax(dim=1)


with torch.no_grad():
    labels_true = []
    labels_predictions_extended = []
    for batch_idx, (images_batch, labels_batch) in enumerate(validation_dataloader):
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
        labels_predictions = model(images_batch).cpu()
        labels_predictions = smax(labels_predictions)
        labels_predictions = numpy.argmax(labels_predictions, axis=1)
        labels_predictions_extended.extend(labels_predictions)
        labels_true.extend(labels_batch.cpu())
        
accuracy = accuracy_score(labels_predictions_extended, labels_true)    
print(f'Validation accuracy: {accuracy}')

matrix = confusion_matrix(labels_true, labels_predictions_extended)
dataframe = pandas.DataFrame(matrix / numpy.sum(matrix) * 10, index = ['0', '1', '2'], columns = ['0', '1', '2'])
plt.figure(figsize = (15, 10))
seaborn.heatmap(dataframe, annot=True, cmap="Purples")
plt.savefig('confusion_matrix.png')


# Compute PREDICTIONS FILE on TEST
with torch.no_grad():
    labels_predictions_extended = []
    for batch_idx, image_batch in enumerate(test_dataloader):
        image_batch = image_batch.to(device)
        label_predictions = model(image_batch).cpu()
        label_predictions = smax(label_predictions)
        label_predictions = numpy.argmax(label_predictions, axis=1)
        labels_predictions_extended.extend(label_predictions)

file = open('../input/ai-unibuc-23-31-2021/test.txt','r')
submission_file = open('./effnet_submission.txt', 'w')
submission_file.write('id,label\n')

for i, line in enumerate(file):
    filename = line.split()[0]
    label = str(labels_predictions_extended[i].item())
    submission_file.write(filename + ',' + label + '\n')

file.close()
submission_file.close()