import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 定义网络结构
# 定义网络结构
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 4*4 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = torch.max_pool2d(torch.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        # x = torch.relu(self.conv2(x))

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.log_softmax(x, dim=1) # 计算log(softmax(x))
        return x



# 定义训练函数
def train(model,epoch,optimizer,train_loader,device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 定义测试函数
def test(model,device):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),  # 图片转为单通道（灰度图）
        torchvision.transforms.ToTensor(),  # PIL Image或者 ndarray 转为tensor，并且做归一化（数据在0~1之间）
        torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化，（均值，标准差）
    ])
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def train_model(train_loader, epochs=1, model=LeNet()):



    # 初始化网络和优化器
    device = torch.device("cpu")
    print(f"Using {device}")
    model = model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        train(model,epoch,optimizer,train_loader,device)
        test(model,device)


    return model

