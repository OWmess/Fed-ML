import syft as sy
import argparse
import torch
import train_mnist
import torchvision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", type=str, default='info@openmined.org', help="Email of the user")
    parser.add_argument("--password", type=str, default='changethis', help="Password of the user")
    parser.add_argument("--port", type=int, default=8081, help="Port of the server")
    parser.add_argument("--save_model", type=bool, required=False, default=False, help="save model or not")
    args = parser.parse_args()

    client = sy.login(email=args.email, password=args.password, port=args.port)

    datasets_all=client.api.services.dataset.get_all()
    print(datasets_all)
    for i in datasets_all:
        print(i)
        for j in i.assets:
            x=i.data
            print(j.data)



