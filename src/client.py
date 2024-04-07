import syft as sy
import argparse
import torch
import train_mnist
import torchvision
from torchvision.datasets import ImageFolder
import pandas as pd
def login_client(email, password, port):
    root_client = sy.login(email=email, password=password, port=port)
    return root_client


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
    parser.add_argument("--email", type=str, default='info@openmined.org',help="Email of the user")
    parser.add_argument("--password", type=str, default='changethis',help="Password of the user")
    parser.add_argument("--port", type=int, default=8081,help="Port of the server")
    parser.add_argument("--client_num", type=int, required=True,help="Client number")
    parser.add_argument("--save_model", type=bool, required=False,default=False, help="save model or not")
    args = parser.parse_args()


    client=sy.login(email=args.email, password=args.password, port=args.port)

    mnist_folder_path = f"../tools/fold_{args.client_num}/"
    train_data = load_mnist(mnist_folder_path)
    model = train_mnist.train_model(train_data,1)
    params=model.state_dict()
    if args.save_model:
        torch.save(params, f"mnist_model_{args.client_num}.pth")

    # 将模型参数及其形状转为一维Numpy数组
    numpy_params = {k: v.numpy().flatten() for k, v in params.items()}
    numpy_shapes = {k: v.shape for k, v in params.items()}

    df_params = pd.DataFrame.from_dict(numpy_params, orient='index')  # 转为Pandas DataFrame
    df_shapes = pd.DataFrame.from_dict(numpy_shapes, orient='index')  # 转为Pandas DataFrame


    dataset = sy.Dataset(
        name=f"client_{args.client_num}_params",
        description=f"MNIST param model from client {args.client_num}",
    )
    dataset.add_contributor(
        role=sy.roles.UPLOADER,
        name="client_1",
        email="alice@openmined.com",
        note="client_1 mnist param",
    )

    asset_mnist_param = sy.Asset(
        name=f"client_{args.client_num}_param",
        description=f"MNIST param model from client {args.client_num}",
        data=df_params,
        mock=sy.ActionObject.empty()
    )

    asset_mnist_shapes=sy.Asset(
        name=f"client_{args.client_num}_param_shapes",
        description=f"MNIST model shapes from client {args.client_num}",
        data=df_shapes,
        mock=sy.ActionObject.empty()
    )

    dataset.add_asset(asset_mnist_param)
    dataset.add_asset(asset_mnist_shapes)
    upload=client.api.services.dataset.upload_dataset(dataset)
    print(upload)
    assert isinstance(upload,sy.SyftError)


    datasets=client.api.services.dataset.get_all()
    print(datasets)