import argparse
import io
import os
import pickle
import socket

import torch
import torchvision
from torchvision.datasets import ImageFolder

import LeNet5

EOT = b'\x7B\x8B\x9B'
STOP_CLIENT_EOT = b'\x0a\x7c\x8b\x9f'


ip = 'localhost'


def load_mnist(path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),  # 图片转为单通道（灰度图）
        torchvision.transforms.ToTensor(),  # PIL Image或者 ndarray 转为tensor，并且做归一化（数据在0~1之间）
        torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化，（均值，标准差）
    ])

    # 从保存的图片中创建数据加载器
    train_data = ImageFolder(root=path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    return train_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_num", type=int, required=True, help="Client number")
    parser.add_argument("--mode", type=str, required=False, help="train mode: iid or non-iid,default idd",
                        default="iid")
    parser.add_argument("--save_model", type=bool, required=False, default=False,
                        help="save model or not,default False")
    parser.add_argument("--ip", type=str, required=False, default='localhost', help="server ip address")
    args = parser.parse_args()
    ip = args.ip

    mnist_folder_path = f"../tools/{args.mode}/fold_{args.client_num}/"
    train_data = load_mnist(mnist_folder_path)
    # 创建socket对象
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接到接收端
    s.connect((ip, 12345))
    while True:

        # 如果服务端有发送数据，接收新的模型参数并加载到模型中
        print('waiting recv...')
        data = b""
        while True:
            packet = s.recv(4096)
            if not packet:
                print('no packet')
                break
            data += packet
            if data.endswith(EOT):
                # print('endswith EOT')
                data = data[:-len(EOT)]
                break
            if data == STOP_CLIENT_EOT:
                if args.save_model:
                    x = torch.rand(1, 1, 28, 28)
                    mod = torch.jit.trace(model, x)
                    if not os.path.exists("../models"):
                        os.makedirs("../models")
                    mod.save(f"../models/mnist_model.pt")
                    torch.save(params, f"../models/mnist_model.pth")
                    # 导出onnx
                    dummy_input = torch.randn(1, 1, 28, 28)
                    torch.onnx.export(model, dummy_input, f"../models/mnist_model.onnx")

                    print(f'trained MNIST model,save at models dir')
                exit(0)
                break
            # 删除终止符
        if data:
            data = pickle.loads(data)
            new_model = data['model']
            buffer = io.BytesIO(new_model)
            model = LeNet5.LeNet()
            model.load_state_dict(torch.load(buffer))
            s.close()

        model = LeNet5.train_model(train_data, 1, model)

        # 浅拷贝，传引用
        params = model.state_dict()
        # if args.save_model:
        #     torch.save(params, f"mnist_model_{args.client_num}.pth")

        # 将模型参数保存到buffer中
        buffer = io.BytesIO()
        torch.save(params, buffer)
        send_struct = {
            'model': buffer.getvalue(),
            'client_id': args.client_num,
        }
        # 序列化数据
        serialized_struct = pickle.dumps(send_struct)
        serialized_struct += EOT
        # 创建socket对象
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 连接到接收端
        s.connect((ip, 12345))
        s.sendall(serialized_struct)
