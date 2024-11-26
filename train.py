import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from Mydatasetnuryourright import Mydataset
import torchvision.transforms.v2 as T


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1
num_epochs = 1
path = osp.normpath(osp.join(osp.dirname(__file__), "data"))

dataset = Mydataset(torch.utils.data.Dataset, get_transform(train=True))
train_set, test_set = torch.utils.data.random_split(dataset, [230, 70])
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False
)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model = model.to(device)

criterion = nn.CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = []

    for batch_idx, (img, targets) in enumerate(train_loader):

        img = img.to(device=device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}]


        # Forward-Pass
        loss_dict = model(img, targets)
        losses = sum(loss for loss in loss_dict.values())  # Summe aller Verluste

        total_loss.append(losses.item())

        # Backward und Optimierung
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f'Cost at epoch {epoch} is {sum(total_loss)/len(total_loss)}')

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in y.items()}]

            scores = model(x)
            predictions = [score['labels'] for score in scores]  # Extract labels from predictions

            for pred, target in zip(predictions, y):
                # Assuming 'labels' is in y
                num_correct += (pred == target['labels']).sum().item()
                num_samples += pred.size(0)

        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct} / {num_samples} correct with accuracy {accuracy:.2f} %")

    model.train()


print("checking accuracy on Training set")
check_accuracy(train_loader, model)
print("checking accuracy on Test set")
check_accuracy(test_loader, model)










