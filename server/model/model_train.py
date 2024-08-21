from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision.transforms import ToTensor
from model.model import CNN

train_data = datasets.MNIST(
    root='./model/data',
    train=True,
    transform=ToTensor(),
    download=True,
)

test_data = datasets.MNIST(
    root='./model/data',
    train=False,
    transform=ToTensor(),
    download=True,
)

train_len = len(train_data)
test_len = len(test_data)

loaders = {
    'train': DataLoader(
            train_data,
            batch_size=128,
            shuffle=True,
            num_workers=4,
    ),
    'test': DataLoader(
            test_data,
            batch_size=128,
            shuffle=True,
            num_workers=4,
    )
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    for epoch in range(1, 11):
        train(epoch)
        test()

    torch.save(model, './model/digit_recognizer_model.pth')