import re
import matplotlib.pyplot as plt
import numpy as np

path = '../charts/'
colors = ['b', 'r', 'm', 'c', 'k']
x = np.arange(50) + 1

def plot_graph(index, param, values, mapper):
    """
    :param index: one of ['PSNR', 'SSIM', 'Loss', 'Val_loss']
    :param param: one of ['first_kernel_size', 'BN', 'n_channels', 'n_res_blocks']
    :param values: values of legend in the graph
    :param mapper: a dictionary in the format: index[legend] = array of values,
            for example: PSNR['with BN'] = array of 50 float values corresponding to 50 epochs.
    :return:
    """
    assert index in ['PSNR', 'SSIM', ['Loss', 'Val_loss']]
    assert param in ['first_kernel_size', 'BN', 'n_channels', 'n_res_blocks']

    if index in ['PSNR', 'SSIM']:
        fig = plt.figure(figsize=(12, 5))
        chart = fig.add_subplot()
        chart.set_xlabel("Epochs")
        chart.set_ylabel(index)
        idx = 0

        for i in values:
            if param == 'first_kernel_size':
                label = r"{0}$\times${0}".format(i)
            elif param == 'n_channels':
                label = r"channels = {}".format(i)
            elif param == 'n_res_blocks':
                label = r"blocks = {}".format(i)
            else:
                label = i
            chart.plot(x, mapper[i], color=colors[idx], label=label)
            idx += 1

        chart.legend()
        fig.savefig("{0}{1}_{2}.pdf".format(path, param, index))
        plt.close(fig)
    else:
        loss = mapper["Loss"]
        val_loss = mapper["Val_loss"]
        idx = 0
        for i in values:
            fig = plt.figure(figsize=(12, 5))
            chart = fig.add_subplot()
            chart.set_xlabel("Epochs")
            chart.set_ylabel("Loss")
            chart.plot(x, loss[i], color='b', label='Training loss')
            chart.plot(x, val_loss[i], color='r', label='Validate loss')
            chart.legend()
            fig.savefig("{0}{1}_{2}_{3}.pdf".format(path, index[idx], param, i))
            plt.close(fig)

def plot(param):
    """
    :param param: one of ['first_kernel_size', 'BN', 'n_channels', 'n_res_blocks']
    :return:
    """
    if param == 'first_kernel_size':
        values = [3, 5, 7, 9, 11]
    elif param == 'n_channels':
        values = [32, 64, 128]
    elif param == 'BN':
        values = ['with BN', 'without BN']
    elif param == 'n_res_blocks':
        values = [3, 5, 7, 9]
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

    indices = ['PSNR', 'SSIM', ['Loss', 'Val_loss']]
    for index in indices:
        if index in ['PSNR', 'SSIM']:
            plot_graph(index, param, values, mapper=PSNR if index == 'PSNR' else SSIM)
        else:
            plot_graph(index, param, values, mapper={'Loss': loss, 'Val_loss': val_loss})

params = ['first_kernel_size', 'BN', 'n_channels', 'n_res_blocks']
for param in params:
    plot(param)



