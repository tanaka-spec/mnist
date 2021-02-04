import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

mnist_train = torchvision.datasets.MNIST(root="mnist_train/", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
train_loader = torch.utils.data.DataLoader(mnist_train)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train(net: Net, epochs: int):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.99, nesterov=True)
	for epoch in range(epochs):
		for (inputs, target) in train_loader:
			optimizer.zero_grad()
			output = net(inputs)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()

def test(net: Net, values):
	correct = 0.0
	total = 0.0
	with torch.no_grad():
		for (inputs, target) in testloader:
			outputs = net(inputs)
			_, predicted = torch.max(outputs.data, 1)
			total += target.size(0)
			correct += (predicted == target).sum().item()
	print('Accuracy: {:.2f} %'.format(correct / total * 100))

if __name__ == '__main__':
	net = Net()
	train(net, 10)
