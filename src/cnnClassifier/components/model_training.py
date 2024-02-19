import os
import urllib.request as request
from zipfile import ZipFile
import torch
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path


class MyLazyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)

class Training(object):
    def __init__(self, config: TrainingConfig):
        # Here we define the attributes of our class
        self.config = config
        self.model = self.get_base_model()
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.params_learning_rate)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Let's send the model to the specified device right away
        self.model.to(self.device)
    
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    
    def get_base_model(self):
        self.model = torch.load(
            self.config.updated_base_model_path
        )
        return self.model

    def set_loaders(self):
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        # Image transformations
        image_transforms = {
            # Train uses data augmentation
            'train':
            transforms.Compose([
                transforms.RandomResizedCrop(size=200, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=64),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
            # Validation does not use augmentation
            'val':
            transforms.Compose([
                transforms.Resize(size=64),
                transforms.CenterCrop(size=180),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        self.dataset = ImageFolder(root=self.config.training_data)
        train, val = torch.utils.data.random_split(self.dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
        if self.config.params_is_augmentation:
            traindataset = MyLazyDataset(train, transform=image_transforms['train'])
            valdataset = MyLazyDataset(val, transform=image_transforms['val'])
        else:
            traindataset = MyLazyDataset(train, transform=image_transforms['val'])
            valdataset = MyLazyDataset(val, transform=image_transforms['val'])
        trainLoader = DataLoader(traindataset , batch_size=self.config.params_batch_size, shuffle=True)
        valLoader = DataLoader(valdataset, batch_size=self.config.params_batch_size)
        
        return trainLoader, valLoader

    def train(self, max_epochs_stop=3,print_every=1, train_on_gpu=True):
        """Train a PyTorch Model

        Params
        --------
            model (PyTorch model): cnn to train
            criterion (PyTorch loss): objective to minimize
            optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
            train_loader (PyTorch dataloader): training dataloader to iterate through
            valid_loader (PyTorch dataloader): validation dataloader used for early stopping
            save_file_name (str ending in '.pt'): file path to save the model state dict
            max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
            n_epochs (int): maximum number of training epochs
            print_every (int): frequency of epochs to print training stats

        Returns
        --------
            model (PyTorch model): trained cnn with best weights
            history (DataFrame): history of train and validation loss and accuracy
        """
        self.set_seed()
        train_loader, valid_loader = self.set_loaders()
        # Early stopping intialization
        epochs_no_improve = 0
        valid_loss_min = np.Inf

        valid_max_acc = 0
        history = []

        # Number of epochs already trained (if using loaded in model weights)
        try:
            print(f'Model has been trained for: {self.model.epochs} epochs.\n')
        except:
            self.model.epochs = 0
            print(f'Starting Training from Scratch.\n')

        overall_start = timer()

        # Main loop
        for epoch in range(1, self.config.params_epochs+1):

            # keep track of training and validation loss each epoch
            train_loss = 0.0
            valid_loss = 0.0

            train_acc = 0
            valid_acc = 0

            # Set to training
            self.model.train()
            start = timer()

            # Training loop
            for ii, (data, target) in enumerate(train_loader):
                # Tensors to gpu
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                # Clear gradients
                self.optimizer.zero_grad()
                # Predicted outputs are log probabilities
                output = self.model(data)

                # Loss and backpropagation of gradients
                loss = self.loss_fn(output, target)
                loss.backward()

                # Update the parameters
                self.optimizer.step()

                # Track train loss by multiplying average loss by number of examples in batch
                train_loss += loss.item() * data.size(0)

                # Calculate accuracy by finding max log probability
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                # Need to convert correct tensor from int to float to average
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples in batch
                train_acc += accuracy.item() * data.size(0)

                # Track training progress
                print(
                    f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                    end='\r')

            # After training loops ends, start validation
            else:
                self.model.epochs += 1

                # Don't need to keep track of gradients
                with torch.no_grad():
                    # Set to evaluation mode
                    self.model.eval()

                    # Validation loop
                    for data, target in valid_loader:
                        # Tensors to gpu
                        if train_on_gpu:
                            data, target = data.cuda(), target.cuda()

                        # Forward pass
                        output = self.model(data)

                        # Validation loss
                        loss = self.loss_fn(output, target)
                        # Multiply average loss times the number of examples in batch
                        valid_loss += loss.item() * data.size(0)

                        # Calculate validation accuracy
                        _, pred = torch.max(output, dim=1)
                        correct_tensor = pred.eq(target.data.view_as(pred))
                        accuracy = torch.mean(
                            correct_tensor.type(torch.FloatTensor))
                        # Multiply average accuracy times the number of examples
                        valid_acc += accuracy.item() * data.size(0)

                    # Calculate average losses
                    train_loss = train_loss / len(train_loader.dataset)
                    valid_loss = valid_loss / len(valid_loader.dataset)

                    # Calculate average accuracy
                    train_acc = train_acc / len(train_loader.dataset)
                    valid_acc = valid_acc / len(valid_loader.dataset)

                    history.append([train_loss, valid_loss, train_acc, valid_acc])

                    # Print training and validation results
                    if (epoch + 1) % print_every == 0:
                        print(
                            f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                        )
                        print(
                            f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                        )

                    # Save the model if validation loss decreases
                    if valid_loss < valid_loss_min:
                        # Save model
                        torch.save(self.model.state_dict(), self.config.trained_model_path)
                        # Track improvement
                        epochs_no_improve = 0
                        valid_loss_min = valid_loss
                        valid_best_acc = valid_acc
                        best_epoch = epoch

                    # Otherwise increment count of epochs with no improvement
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= max_epochs_stop:
                            print(
                                f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch+1} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                            )
                            total_time = timer() - overall_start
                            print(
                                f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
                            )

                            # Load the best state dict
                            self.model.load_state_dict(torch.load(self.config.trained_model_path))
                            # Attach the optimizer
                            self.model.optimizer = self.optimizer

                            # Format history
                            history = pd.DataFrame(
                                history,
                                columns=[
                                    'train_loss', 'valid_loss', 'train_acc',
                                    'valid_acc'
                                ])
                            return self.model, history

        # Attach the optimizer
        self.model.optimizer = self.optimizer
        # Record overall time and print out stats
        total_time = timer() - overall_start
        print(
            f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
        )
        print(
            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
        )
        # Format history
        history = pd.DataFrame(
            history,
            columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
        
        self.save_checkpoint(self.model)
        return self.model, history
    
    def save_checkpoint(self, model, multi_gpu=False):
        """Save a PyTorch model checkpoint

        Params
        --------
            model (PyTorch model): model to save
            path (str): location to save model. Must start with `model_name-` and end in '.pth'

        Returns
        --------
            None, save the `model` to `path`

        """

        # Basic details
        checkpoint = {
            'epochs': model.epochs
        }

        # Extract the final classifier and the state dictionary
        # Check to see if model was parallelized
        if multi_gpu:
            checkpoint['classifier'] = model.module.classifier
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['classifier'] = model.classifier
            checkpoint['state_dict'] = model.state_dict()

        # Add the optimizer
        checkpoint['optimizer'] = model.optimizer
        checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

        # Save the data to the path
        torch.save(checkpoint, self.config.trained_model_path)