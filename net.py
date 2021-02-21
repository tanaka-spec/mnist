import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64 
        self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train(net: Net, epochs: int, batch_size: int, mnist_train):
	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.99, nesterov=True)
	print('Training with Epoch: {:d}'.format(epochs))
	for epoch in range(epochs):
		for (inputs, target) in train_loader:
			optimizer.zero_grad()
			output = net(inputs)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()

def test(net: Net, batch_size: int, mnist_test):
	test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=0)
	correct = 0.0
	total = 0.0
	with torch.no_grad():
		for (inputs, target) in test_loader:
			outputs = net(inputs)
			_, predicted = torch.max(outputs.data, 1)
			total += target.size(0)
			correct += (predicted == target).sum().item()
	print('Test Accuracy: {:.2f} %'.format(correct / total * 100))

def eval(net: Net, batch_size: int, mnist_eval):
	eval_loader = torch.utils.data.DataLoader(mnist_eval, batch_size=batch_size, shuffle=True, num_workers=0)
	correct = 0.0
	total = 0.0
	with torch.no_grad():
		for (inputs, target) in eval_loader:
			outputs = net(inputs)
			_, predicted = torch.max(outputs.data, 1)
			total += target.size(0)
			correct += (predicted == target).sum().item()
	print('Evaluation Accuracy: {:.2f} %, Batch size: {:d}'.format(correct / total * 100, batch_size))
	return correct / total * 100

def main():
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
	mnist_data = torchvision.datasets.MNIST(root="mnist_train/", train=True, download=True, transform=transform)
	mnist_train, mnist_eval = torch.utils.data.random_split(mnist_data, [int(len(mnist_data) * 0.75), len(mnist_data) - int(len(mnist_data) * 0.75)])
	mnist_test = torchvision.datasets.MNIST(root="mnist_test/", train=False, download=True, transform=transform)

	net = Net()
	epoch = random.randint(1, 5)
	batch_size = random.randint(10, 1000)
	accuracy = 0
	while (accuracy < 90):
		train(net, epoch, batch_size, mnist_train)
		accuracy = eval(net, batch_size, mnist_eval)
		epoch = epoch + 1
	test(net, batch_size, mnist_test)


if __name__ == '__main__':
	main()
