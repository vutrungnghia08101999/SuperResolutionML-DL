import re
import matplotlib.pyplot as plt
import numpy as np

path = '../charts/'
def plot(param): # one of ['depth', 'BN', 'kernel_size', 'ReLU_Tanh']
    if param == 'depth':
        values = [2, 4, 6, 8]
    elif param == 'kernel_size':
        values = [3, 5, 7, 9]
    elif param == 'BN':
        values = ['without BN', 'with BN']
    elif param == 'ReLU_Tanh':
        values = ['ReLU', 'Tanh']
    else:
        print("Param not valid")
        return -1

    input_file = "logs_{}.txt".format(param)
    with open("../{}".format(input_file), "r") as fr:
        raw = fr.read()

    data = re.findall(r"EPOCH\W+(\d+)\W+Loss\W+([\d\.]+)(?s:.*?)Val_loss\W+([\d\.]+)\W+PSNR\W+([\d\.]+)\W+SSIM\W+([\d\.]+)", raw)
    dt = np.dtype([('EPOCH', 'int'), ('Val_loss', 'float'), ('Loss', 'float'), ('PSNR', 'float'), ('SSIM', 'float')])
    data = np.array(data, dtype=dt)
    rows = len(data) // 50
    data.shape = (rows, 50)

    PSNR = {}
    SSIM = {}
    loss = {}
    val_loss = {}
    idx = 0
    for i in values:
        PSNR[i] = np.array([x["PSNR"] for x in data[idx]])
        SSIM[i] = np.array([x["SSIM"] for x in data[idx]])
        loss[i] = np.array([x["Loss"] for x in data[idx]])
        val_loss[i] = np.array([x["Val_loss"] for x in data[idx]])
        idx += 1

    x = np.arange(50) + 1

    colors = ['b', 'r', 'm', 'c']

    psnr = plt.figure(figsize=(12, 5))
    psnr_chart = psnr.add_subplot()
    psnr_chart.set_xlabel("Epochs")
    psnr_chart.set_ylabel("PSNR")
    idx = 0

    for i in values:
        if param == 'kernel_size':
            label = r"{0}$\times${1}".format(i, i)
        elif param == 'depth':
            label = 'depth = {}'.format(i)
        else:
            label = i
        plt.plot(x, PSNR[i], color=colors[idx], label=label)
        idx += 1

    psnr_chart.legend()
    psnr.savefig("{0}{1}_PSNR.pdf".format(path, param))

    psnr = plt.figure(figsize=(12, 5))
    psnr_chart = psnr.add_subplot()
    psnr_chart.set_xlabel("Epochs")
    psnr_chart.set_ylabel("SSIM")
    idx = 0

    for i in values:
        if param == 'kernel_size':
            label = r"{0}$\times${1}".format(i, i)
        elif param == 'depth':
            label = 'depth = {}'.format(i)
        else:
            label = i
        plt.plot(x, SSIM[i], color=colors[idx], label=label)
        idx += 1

    psnr_chart.legend()
    psnr.savefig("{0}{1}_SSIM.pdf".format(path, param))
    
params = ['depth', 'BN', 'kernel_size', 'ReLU_Tanh']
for param in params:
    plot(param)



