import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
# 定义网络结构
# 定义网络结构
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # 最大池化为（2，2）
        x = torch.max_pool2d(torch.relu(self.conv1(x)), (2, 2))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # 展平除批量维度之外的所有维度
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

    # 初始化列表以保存真实标签和预测标签
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # 添加真实标签和预测标签到列表中
            all_targets.extend(target.view_as(pred).cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

    test_loss /= len(test_loader.dataset)

    cm = confusion_matrix(all_targets, all_preds)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss , 100. * correct / len(test_loader.dataset),cm

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

