import io
import socket
import torch
import pickle
import threading
import train_mnist
import time
import os
from collections import OrderedDict
import visdom
EOT=b'\x7B\x8B\x9B'
STOP_CLIENT_EOT=b'\x0a\x7c\x8b\x9f'
CLIENT_NUM=1
# 全局字典，用于保存接收到的模型
models = {}
clients= {}

# 记录迭代次数
iteration = 1
success_cnt=0
exit_flag=False

vis=visdom.Visdom()


def handle_client(conn, addr):
    # 读取数据
    data = b""
    while True:
        packet = conn.recv(4096)
        if not packet:
            print('no packet')
            break
        data += packet
        if data.endswith(EOT):
            print('endswith EOT')
            data = data[:-len(EOT)]
            break
        # 删除终止符


    # 反序列化数据
    data = pickle.loads(data)

    # 提取模型和元数据
    model_bytes = data['model']
    client_id = data['client_id']

    # 加载模型
    buffer = io.BytesIO(model_bytes)
    model = train_mnist.LeNet()
    model.load_state_dict(torch.load(buffer))

    # 打印元数据
    print('Received model from client {} '.format(client_id))

    # 保存模型到全局字典

    models[client_id] = model
    clients[client_id] = conn
    # 测试模型部分
    # if client_id in models:
    #     model = models[client_id]
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model=model.to(device)
    #     train_mnist.test(model,device)


def federated_avg(models):
    # 创建一个字典来存储平均权重
    avg_weights = {}

    # 遍历模型的权重
    for model in models.values():
        model_weights = model.state_dict()

        # 如果是第一个模型，复制其权重
        if not avg_weights:
            avg_weights = model_weights
        else:
            # 如果不是第一个模型，将权重相加
            for key in model_weights:
                avg_weights[key] += model_weights[key]

    # 除以模型的数量来获取平均权重
    for key in avg_weights:
        avg_weights[key] /= CLIENT_NUM

    return avg_weights

def send_model(conn, model):
    # 序列化模型
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    # 创建数据字典
    data = {'model': buffer.getvalue()}

    # 将数据字典序列化
    data_bytes = pickle.dumps(data)
    data_bytes += EOT
    # 发送数据
    conn.sendall(data_bytes)
    print('Send FedAvg param to client.')

def stop_work(conn):
    conn.sendall(STOP_CLIENT_EOT)
    pass

def check_models():
    global iteration
    global success_cnt
    while True:
        # 检查models字典是否包含从0到4的所有客户端ID
        if all(i in models for i in range(CLIENT_NUM)):
            print("Models from all clients have been received.")
            avg_weight=federated_avg(models)
            model=train_mnist.LeNet()
            model.load_state_dict(avg_weight)
            device=torch.device("cpu")
            model=model.to(device)
            print(f"epoch: {iteration}：")
            loss,accuracy=train_mnist.test(model,device)
            vis.line(X=[iteration], Y=[accuracy], win='accuracy', \
                     update='append' if iteration > 0 else None, opts=dict(title='accuracy'))
            vis.line(X=[iteration], Y=[loss], win='loss', \
                        update='append' if iteration > 0 else None, opts=dict(title='loss'))
            if success_cnt >= 3:
                print('Training completed.')
                for conn in clients.values():
                    stop_work(conn)

                os._exit(0)
            if accuracy > 98:
                success_cnt+=1
            # 清空models字典
            iteration += 1
            models.clear()
            # 将新的模型权重发送到每个客户端
            for conn in clients.values():
                send_model(conn, model)
            x = torch.rand(1, 1, 28, 28)
            mod = torch.jit.trace(model, x)
            if not os.path.exists("../models"):
                os.makedirs("../models")
            mod.save(f"../models/mnist_model.pt")
            params=model.state_dict()
            torch.save(params, f"../models/mnist_model.pth")
            # 导出onnx
            dummy_input = torch.randn(1, 1, 28, 28)
            torch.onnx.export(model, dummy_input, f"../models/mnist_model.onnx")
            # 关闭所有的连接并清空客户端连接字典
            # for conn in clients.values():
            #     conn.close()
            clients.clear()
        # 在再次检查之前等待一段时间
        time.sleep(0.5)  # 你可以根据需求调整这个休眠时间




if __name__ == "__main__":
    print("Fed_ml server start.")
    # 创建socket对象
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定到特定的地址和端口
    s.bind(('localhost', 12345))

    # 开始监听连接
    s.listen(1)

    check_thread = threading.Thread(target=check_models)
    check_thread.start()
    # 服务器主循环
    while True:
        # 接受连接
        conn, addr = s.accept()

        # 创建一个新的线程来处理这个连接
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
