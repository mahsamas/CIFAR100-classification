import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

print("imported everything... I think")

# device thing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using >>>", device)

# transforms for images
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
])

# datasets
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

print("datasets ready, train size:", len(trainset), "test size:", len(testset))

# make a simple cnn (not fancy, just something)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc2 = nn.Linear(256, 100)  # 100 classes for CIFAR100
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64*8*8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net().to(device)

# loss + optimizer
crit = nn.CrossEntropyLoss()
opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# training loop
epochs = 5
for e in range(epochs):
    print("epoch:", e+1)
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        outputs = net(inputs)
        loss = crit(outputs, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 0: # just some random print
            print("batch", i, "loss:", loss.item())
    print("epoch loss:", running_loss/len(trainloader))
    print("train acc:", 100*correct/total, "%")

print("training done")

# test loop
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        xx, yy = data
        xx, yy = xx.to(device), yy.to(device)
        out = net(xx)
        _, preds = torch.max(out, 1)
        total += yy.size(0)
        correct += (preds == yy).sum().item()

print("Final Test Acc:", 100*correct/total, "%")
