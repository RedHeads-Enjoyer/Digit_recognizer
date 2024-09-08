import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(
        degrees=10,  # increased rotation range
        scale=(0.9, 1.1)  # increased scale range (small to large)
    ),
])

train_data = datasets.MNIST(
    root='./model/data',
    train=True,
    transform=transform,
    download=True,
)

test_data = datasets.MNIST(
    root='./model/data',
    train=False,
    transform=transform,
    download=True,
)


train_len = len(train_data)
test_len = len(test_data)

loaders = {
    'train': DataLoader(
            train_data,
            batch_size=100,
            shuffle=True,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
    ),
    'test': DataLoader(
            test_data,
            batch_size=100,
            shuffle=True,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
    )
}

class CNN(nn.Module):
    def calculate_output_shape(self):
        x = torch.randn(1, 1, 28, 28)  # assuming input shape is (1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        return x.size(1) * x.size(2) * x.size(3)

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        output_shape = self.calculate_output_shape()
        self.fc1 = nn.Linear(output_shape, 320)
        self.fc2 = nn.Linear(320, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))  # Added ReLU activation
        x = self.fc3(x)
        return F.softmax(x, dim=1)

def load_model():
    _model = torch.load('./model/digit_recognizer_model.pth', weights_only=False)
    _model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    _model.eval()
    return _model


model = CNN().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{train_len} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\t{loss.item():.6f}')

def test():
    model.eval()

    test_loss = 0
    correct = 0

    with torch.inference_mode():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_len
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{test_len} ({100. * correct / test_len :.0f}%)\n")


def save_model():
    for epoch in range(1, 6):
        train(epoch)
        test()
    torch.save(model, f'./model/digit_recognizer_model.pth')

