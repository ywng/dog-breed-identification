
# coding: utf-8

# In[72]:

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle


# In[73]:

num_classes = 120
batch_size = 5
epochs = 15
sample_submission = pd.read_csv('./data/sample_submission.csv')


# In[74]:

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_transform = transforms.Compose([transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
valid_transform = transforms.Compose([transforms.Resize(360),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                normalize])


trainset = torchvision.datasets.ImageFolder("./data/train/", train_transform);
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

validset = torchvision.datasets.ImageFolder("./data/valid/", valid_transform);
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

classes = trainloader.dataset.classes
pickle.dump(classes, open("dog_breeds_labels.pickle", "wb"), 2)


# In[75]:

def train(model, train, valid, optimizer, criterion, epochs=1):
    for epoch in range(epochs):
        print('Epoch ', epoch + 1, '/', epochs)
        
        running_loss = 0.
        running_corrects = 0.
        running_batches = 0.
       
        model.train()
        for i, (input, target) in enumerate(train):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            optimizer.zero_grad()

            output = model(input_var)
            if type(output) == tuple:
                output, _ = output
            _, preds = torch.max(output.data, 1)
            loss = criterion(output, target_var)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0]
            running_corrects = preds.eq(target_var.data).cpu().sum()
            running_batches += 1.
            

            print('\r', 'Batch', i, 'Loss', loss.data[0], end='')
            
        train_loss = running_loss / running_batches
        train_acc = running_corrects / len(train.dataset)
        print('\r', "Train Loss", train_loss, "Train Accuracy", train_acc)
            
        running_loss = 0.
        running_corrects = 0.
        running_batches = 0.

        model.eval()
        for i, (input, target) in enumerate(valid):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            output = model(input_var)
            if type(output) == tuple:
                output, _ = output
            _, preds = torch.max(output.data, 1)
            loss = criterion(output, target_var)

            running_loss += loss.data[0]
            running_corrects = preds.eq(target_var.data).cpu().sum()
            running_batches += 1.
            

        valid_loss = running_loss / running_batches
        valid_acc = running_corrects / len(valid.dataset)
        print('\r', "Val Loss", valid_loss, "Val Accuracy", valid_acc)


# In[76]:

model = torchvision.models.inception_v3(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
num_features = model.fc.in_features
#replace the classifier of the trained network with our dog classifier
#parameters of newly constructed modules have required_grad=True by default
model.fc = nn.Linear(num_features, num_classes) 

ct =  []
for name, child in model.named_children():
    if "Conv2d_4a_3x3" in ct:
        for params in child.parameters():
            params.requires_grad = True
    ct.append(name)
    
criterion = nn.CrossEntropyLoss()
#Optimize only the classifier
optimizer = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001, momentum=0.9)

if torch.cuda.is_available():
    print("Cuda is available.")
    model = torch.nn.DataParallel(model).cuda();
    
train(model, trainloader, validloader, optimizer, criterion, epochs=epochs)


# In[ ]:

torch.save(model.state_dict(), 'dog-breed-ident-inceptionv3-GPU.pt')


# In[ ]:

class DogsData(Dataset):
    def __init__(self, root_dir, labels, transform, output_class=True):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.output_class = output_class
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = self.labels.iloc[idx]
        path = '{}/{}.jpg'.format(self.root_dir, item['id'])
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        if not self.output_class:
            return image
        
        return image, item['class']

test_loader = torch.utils.data.DataLoader(
    DogsData('./data/test/x', sample_submission, valid_transform, output_class=False),
    batch_size=batch_size
)


# In[ ]:

model.load_state_dict(torch.load('dog-breed-ident-inceptionv3-GPU.pt'))
model.eval()
results = []

for i, input in enumerate(test_loader):
    input_var = torch.autograd.Variable(input, volatile=True)
    
    if torch.cuda.is_available():
        input_var = input_var.cuda()
        
    output = model(input_var)
    results.append(F.softmax(output).cpu().data.numpy())
    print('\r', 'Batch', i, end='')
    
    ''' for testing 
    if i > 10:
        break
    '''
        
results = np.concatenate(results)


# In[ ]:

results.shape


# In[ ]:

ids = sample_submission['id'].values
sample_df = pd.DataFrame(ids, columns=['id'])
#sample_df = sample_df[:48] #for testing only
#sample_df.shape
for index, breed in enumerate(classes):
    sample_df[breed] = results[:,index]


# In[ ]:

sample_df.to_csv('pred_inceptionv3.csv', index=None)


# In[ ]:



