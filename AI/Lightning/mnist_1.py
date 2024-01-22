import os
import torch
from lightning import LightningModule, Trainer, LightningDataModule
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelSummary

MAX_EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 7


class MNISTDataModule(LightningDataModule):
    """
    Used to load data (28x28x1 images of handwritten digits) from MNIST dataset 
    """

    def setup(self, stage):
        # used to transform an image into Tensor
        transform=transforms.ToTensor()
        # prepare transforms standard to MNIST
        self.train_set = MNIST(os.getcwd(), train=True, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)




class MNISTModel(LightningModule):
    """
    Used to build a Neural Network and to train it with data (images) from a MNISTDataModule
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(28*28, hidden_dim)  # connect the 28x28 pixel inputs to the 'hidden_dim' neurons of the hidden layer
        self.l2 = torch.nn.Linear(hidden_dim, 10)  # connect the 'hidden_dim' neurons of the hidden layer to the 10 digit outputs

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
