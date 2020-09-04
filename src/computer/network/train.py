"""
Performs the training
"""
import argparse
import torch
import torch.nn as nn
import os
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple

from training_model import TrainingNet
from dataset import Dataset
from utils import generate_confusion_matrix

class Train():
    
    def __init__(self, data_path: Path, checkpoint_path: Path, logdir_path: Path):
        """Initializes the training

        Args:
            data_path (Path): Path to data
            checkpoint_path (Path): Path to the checkpoint directory
            logdir_path (Path): Path to the log directory
        """        
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.logdir_path = logdir_path

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

    def train_network(self, batch_size: int, learning_rate: float, epochs: int, checkpoint: Path = None) -> None:
        """Starts the training

        Args:
            batch_size (int): Batch size used for training
            learning_rate (float): Learning rate used for training
            epochs (int): Number of epochs to train
            checkpoint (Path, optional): Path to a pretrained model. Defaults to None.
        """        

        sub_datasets = ['train', 'val', 'test']
        datasets = {x: Dataset((self.data_path / x), train=(x == 'train')) for x in sub_datasets}

        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}
        print(dataset_sizes)

        dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x == 'train')) for x in sub_datasets}

        target_classes = datasets['train'].get_target_dict()

        NET = TrainingNet(num_classes=len(target_classes))

        run_name = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
        log_name = self.logdir_path / run_name

        writer = SummaryWriter(log_name.absolute().as_posix(), flush_secs=30)

        if checkpoint:
            NET.load_state_dict(torch.load(self.checkpoint_path / checkpoint))

        NET.to(self.device)

        opt = torch.optim.SGD(NET.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.3)

        NET.train()
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        model_path = self.checkpoint_path / run_name


        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)

            training_loss = 0
            training_accuracy = 0

            t0 = time.time()

            # training
            NET.train()
            for batch in dataloaders["train"]:
                opt.zero_grad()

                image = batch["image"].to(self.device)
                target = batch["target"].to(self.device)
                condition = batch["condition"].to(self.device)
                
                outputs = NET(image, condition)

                loss = criterion(outputs, target)

                loss.backward()
                opt.step()

                training_loss += loss.item() * image.size(0)
                _, preds = torch.max(outputs, 1)
                training_accuracy += torch.sum(preds == target.data)

            training_loss = training_loss / dataset_sizes["train"]
            training_accuracy = training_accuracy.double() / dataset_sizes["train"]

            writer.add_scalar('Loss/train', training_loss, epoch)
            writer.add_scalar('Accuracy/train', training_accuracy, epoch)

            print('Train: {:.1f} seconds'.format(time.time() - t0))

            t0 = time.time()

            # validation
            real_values, predictions, validation_loss, validation_accuracy = self._evaluate(dataloaders["val"], NET, criterion)

            scheduler.step()

            writer.add_scalar('Loss/val', validation_loss, epoch)
            writer.add_scalar('Accuracy/val', validation_accuracy, epoch)

            if validation_accuracy > best_acc:
                print("new val_acc: {}".format(validation_accuracy))
                best_acc = validation_accuracy

                torch.save(NET.state_dict(), model_path.absolute().as_posix())

            if epoch % 5 == 0 or epoch == epochs - 1:
                generate_confusion_matrix(real_values, predictions, writer)

            print('Val: {:.1f} seconds'.format(time.time() - t0))
            print()

            t0 = time.time()

        # end training, save model
        writer.close()
        model_path = model_path.with_name(run_name + "_last")
        torch.save(NET.state_dict(), model_path.absolute().as_posix())

    def _evaluate(
        self, 
        dataloaders: DataLoader, 
        NET: nn.Module, criterion: 
        nn.modules.loss
        ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """Evaluates the network

        Args:
            dataloaders (DataLoader): Dataloader of dataset to evaluate
            NET (nn.Module): Network model
            criterion (nn.modules.loss): Loss criterion to calculate loss

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]: 
                Tuple containing tensors with the correct and the 
                predicted labels as well as the evaluation loss and accuracy
        """        
        
        predictions = []
        real_values = []
        loss = 0
        accuracy = 0
        dataset_size = len(dataloaders.dataset)

        NET.eval()
        with torch.no_grad():
            for batch in dataloaders:
                image = batch["image"].to(self.device)
                target = batch["target"].to(self.device)
                condition = batch["condition"].to(self.device)
                
                outputs = NET(image, condition)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds)
                real_values.extend(target)

                ls = criterion(outputs, target)

                loss += ls.item() * image.size(0)
                accuracy += torch.sum(preds == target.data)

        predictions = torch.as_tensor(predictions).cpu()
        real_values = torch.as_tensor(real_values).cpu()

        
        loss = loss / dataset_size
        accuracy = accuracy.double() / dataset_size

        return real_values, predictions, loss, accuracy

        