import os
import torch
from lightning import LightningModule, Trainer, LightningDataModule
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from lightning.pytorch.callbacks import ModelSummary


MAX_EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 7
NUM_CLASSES = 10 # All the ten digits


class MNISTDataModule(LightningDataModule):
    """
    Used to load data (28x28x1 images of handwritten digits) from MNIST dataset 
    """

    def setup(self, stage):
        # used to transform an image into Tensor
        transform=transforms.ToTensor()
        # prepare transforms standard to MNIST
        train_and_val = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        # divide into validation and training set
        self.train_set, self.val_set= random_split(train_and_val,[55000,5000])
        self.test_set = MNIST(os.getcwd(), train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)



class MNISTModel(LightningModule):
    """
    Used to build a Neural Network and to train it with data (images) from a MNISTDataModule
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(28*28, hidden_dim)  # connect the 28x28 pixel inputs to the 'hidden_dim' neurons of the hidden layer
        self.l2 = torch.nn.Linear(hidden_dim, 10)  # connect the 'hidden_dim' neurons of the hidden layer to the 10 digit outputs
        self.example_input_array = torch.Tensor(BATCH_SIZE, 1, 28, 28)  # To print input output layer dimensions

    def forward(self, x):
        """Process a forward pass with an image as input and outputs 10 probabilities"""
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)  # Flatten the x tensor
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("Train_loss",loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task='multiclass', num_classes=NUM_CLASSES)
        self.log_dict({"Val_loss":loss, "Val_acc": acc}, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task='multiclass', num_classes=NUM_CLASSES)
        self.log_dict({"Test_loss":loss, "Test_acc": acc}, on_epoch=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    

# Init our model
mnist_model = MNISTModel(hidden_dim=300)

# Init DataLoader from MNIST Dataset
data_module = MNISTDataModule()

# Initialize a trainer
trainer = Trainer(max_epochs=MAX_EPOCHS, callbacks=[ModelSummary(max_depth=-1)], fast_dev_run=False)

# To visualize the model with Netron. See : https://machinelearningmastery.com/visualizing-a-pytorch-model/
X_test = torch.Tensor(BATCH_SIZE, 1, 28, 28)  # Fake data, just to be processed by the model
torch.onnx.export(mnist_model, X_test, 'model.onnx', input_names=["features"], output_names=["logits"])

# Train the model
trainer.fit(mnist_model, data_module)

# Test the model 
trainer.test(mnist_model, data_module)
