import torch
from my_eeg_dataset import MySeizureDataset
from torch.utils.data import DataLoader
import os
from config import Config, default_config
import torchvision.models as models
from my_models import my_cnn_model
import torch.nn as nn
from torch.autograd import Variable
import random


class CombinedSeizureDataset(torch.utils.data.Dataset):
    def __init__(self, file_pairs):
        """
        Dataset that combines multiple EDF files
        file_pairs: List of tuples (edf_file, label_file)
        """
        self.datasets = []
        for edf_file, label_file in file_pairs:
            dataset = MySeizureDataset(edf_file, label_file)
            self.datasets.append(dataset)
        
        # Create index mapping
        self.dataset_idx = []
        for i, dataset in enumerate(self.datasets):
            for j in range(len(dataset)):
                self.dataset_idx.append((i, j))
    
    def __getitem__(self, index):
        dataset_id, sample_id = self.dataset_idx[index]
        return self.datasets[dataset_id][sample_id]
    
    def __len__(self):
        return len(self.dataset_idx)


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

def load_and_split_data(config, split_ratio=(0.7, 0.15, 0.15)):
    """
    Load all EDF files and split them into train, validation, and test sets
    
    Args:
        config: Configuration object
        split_ratio: Tuple of (train_ratio, val_ratio, test_ratio)
        
    Returns:
        train_files, val_files, test_files: Lists of file pairs
    """
    # Collect all .edf files from train_cases
    all_edf_files = []
    for case in config.train_cases:
        data_dir = os.path.join(config.data_root, case)
        if not os.path.exists(data_dir):
            raise Exception(f"Error: Data directory {data_dir} does not exist.")
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".edf"):
                    edf_file = os.path.join(root, file)
                    label_file = edf_file.replace("eeg.edf", "events.tsv")
                    all_edf_files.append((edf_file, label_file))
    
    # Split files into training, validation, and test sets
    num_files = len(all_edf_files)
    num_train = int(split_ratio[0] * num_files)
    num_val = int(split_ratio[1] * num_files)
    
    # Shuffle the files randomly
    random.shuffle(all_edf_files)
    
    train_files = all_edf_files[:num_train]
    val_files = all_edf_files[num_train:num_train+num_val]
    test_files = all_edf_files[num_train+num_val:]
    
    print(f"Total files: {num_files}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    return train_files, val_files, test_files

def train_epoch(model, train_loader, optimizer, error, epoch):
    """
    Train the model for one epoch
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        error: Loss function
        epoch: Current epoch number
        
    Returns:
        train_loss, train_accuracy: Average loss and accuracy for the epoch
    """
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = torch.unsqueeze(X_batch, 1)
        var_X_batch = Variable(X_batch).float().cuda()
        var_y_batch = Variable(y_batch).cuda()
        
        optimizer.zero_grad()
        output = model(var_X_batch)
        loss = error(output.cuda(), var_y_batch.type(torch.LongTensor).cuda())
        loss.backward()
        optimizer.step()
        
        # Track loss
        train_loss += loss.item()
        
        # Total correct predictions
        predicted = torch.max(output.data, 1)[1] 
        train_correct += (predicted == var_y_batch).sum()
        train_total += y_batch.size(0)
        
        if batch_idx % 50 == 0:
            print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                epoch, batch_idx*len(X_batch), len(train_loader.dataset), 
                100.*batch_idx / len(train_loader), loss.data, 
                float(train_correct*100) / float(train_total)))
    
    # Calculate average loss and accuracy
    avg_loss = train_loss / len(train_loader)
    accuracy = float(train_correct*100) / float(train_total)
    
    return avg_loss, accuracy

def validate(model, val_loader, error):
    """
    Validate the model on the validation set
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        error: Loss function
        
    Returns:
        val_loss, val_accuracy: Average loss and accuracy for validation
    """
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = torch.unsqueeze(X_batch, 1)
            var_X_batch = Variable(X_batch).float().cuda()
            var_y_batch = Variable(y_batch).cuda()
            
            output = model(var_X_batch)
            val_loss += error(output.cuda(), var_y_batch.type(torch.LongTensor).cuda())
            
            predicted = torch.max(output.data, 1)[1]
            val_correct += (predicted == var_y_batch).sum()
            val_total += y_batch.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = val_loss / len(val_loader)
    accuracy = float(val_correct*100) / float(val_total)
    
    return avg_loss, accuracy

def fit(model, config):
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    EPOCHS = config.n_epochs
    
    # Load and split data
    train_files, val_files, test_files = load_and_split_data(config)
    
    # Create datasets and dataloaders
    train_dataset = CombinedSeizureDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    val_dataset = CombinedSeizureDataset(val_files)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Training loop
    for epoch in range(EPOCHS):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, error, epoch)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, error)
        
        print('Validation Epoch: {} \tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
            epoch, val_loss, val_acc))
                
if __name__ == "__main__":
    train_cases = ['sub-00', 'sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07', 'sub-09','sub-10', 'sub-11', 'sub-12']
    test_cases = ['sub-13', 'sub-14', 'sub-16', 'sub-17']
    


    model, config = create_model()
    fit(model.cuda(), config)


    print("Done")


