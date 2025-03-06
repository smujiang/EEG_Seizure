import torch
from my_eeg_dataset import MySeizureDataset
from torch.utils.data import DataLoader
import os
from config import Config, default_config
import torchvision.models as models
from my_models import my_cnn_model
import torch.nn as nn
from torch.autograd import Variable


def toy_train(model, config):
    for epoch in range(config.n_epochs):
        for case in config.train_cases:
            data_dir = os.path.join(config.data_root, case)
            if not os.path.exists(data_dir):
                raise Exception("Error: Data directory does not exist.")
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(".edf"):
                        print(file)
                        edf_file = os.path.join(root, file)
                        label_file = edf_file.replace("eeg.edf", "events.tsv")
                        eeg_dataset = MySeizureDataset(edf_file, label_file)
                        dataloader = DataLoader(eeg_dataset, batch_size=config.batch_size, shuffle=True)
                        for i, (x, y) in enumerate(dataloader):
                            # train model
                            model.train()
        # calculate loss
        # write log

def create_model():
    config = Config(default_config) # default_config is from config.py
    config.batch_size = 128  # what if batch_size = 8
    config.n_epochs = 2000 # number of epochs
    config.c_dim = 2  # number of classes
    config.lr = 0.0002 # learning rate
    config.data_root = "/data/jjiang10/Data/EEG/BIDS_Siena"
    config.train_cases = ['sub-00', 'sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07', 'sub-09','sub-10', 'sub-11', 'sub-12']

    model = my_cnn_model()
    
    return model, config

def fit(model, config):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = config.n_epochs
    model.train()
    for epoch in range(EPOCHS):
        for case in config.train_cases:
            
            data_dir = os.path.join(config.data_root, case)
            if not os.path.exists(data_dir):
                raise Exception("Error: Data directory does not exist.")
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(".edf"):
                        print(file)
                        edf_file = os.path.join(root, file)
                        label_file = edf_file.replace("eeg.edf", "events.tsv")
                        eeg_dataset = MySeizureDataset(edf_file, label_file)
                        train_loader = DataLoader(eeg_dataset, batch_size=config.batch_size, shuffle=True)
                        correct = 0
                        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                            # print(X_batch.shape)
                            X_batch = torch.unsqueeze(X_batch, 1)
                            # print(X_batch.shape)
                            var_X_batch = Variable(X_batch).float().cuda()
                            var_y_batch = Variable(y_batch).cuda()
                            optimizer.zero_grad()
                            output = model(var_X_batch)
                            loss = error(output.cuda(), var_y_batch.type(torch.LongTensor).cuda())
                            loss.backward()
                            optimizer.step()

                            # Total correct predictions
                            predicted = torch.max(output.data, 1)[1] 
                            correct += (predicted == var_y_batch).sum()
                            #print(correct)
                            if batch_idx % 50 == 0:
                                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.data, float(correct*100) / float(config.batch_size*(batch_idx+1))))
                                
if __name__ == "__main__":
    train_cases = ['sub-00', 'sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07', 'sub-09','sub-10', 'sub-11', 'sub-12']
    test_cases = ['sub-13', 'sub-14', 'sub-16', 'sub-17']
    


    model, config = create_model()
    fit(model.cuda(), config)


    print("Done")


