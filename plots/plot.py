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
    assert index in ['PSNR', 'SSIM', 'D_loss', 'G_loss', ['sr_prob', 'hr_prob'], ['SR_PROB', 'HR_PROB']]
    assert param in ['D_first_kernel_size', 'n_blocks', 'classifier', 'BN']

    # if index in ['PSNR', 'SSIM', 'sr_prob', 'SR_PROB', 'hr_prob', 'HR_PROB']:
    if index in ['PSNR', 'SSIM']:
        fig = plt.figure(figsize=(12, 5))
        chart = fig.add_subplot()
        chart.set_xlabel("Epochs")
        # if index == 'sr_prob':
        #     ylabel = 'Train SR probability'
        # elif index == 'SR_PROB':
        #     ylabel = 'Validate SR probability'
        # elif index == 'hr_prob':
        #     ylabel = 'Train HR probability'
        # elif index  == 'HR_PROB':
        #     ylabel = 'Validate HR probability'
        # else: ylabel = index

        chart.set_ylabel(index)
        idx = 0

        for i in values:
            if param == 'D_first_kernel_size':
                label = r"{0}$\times${0}".format(i)
            else:
                label = i
            chart.plot(x, mapper[i], color=colors[idx], label=label)
            idx += 1

        chart.legend()
        fig.savefig("{0}{1}_{2}.pdf".format(path, param, index))
        plt.close(fig)

    elif index in ['G_loss', 'D_loss']:

        if index == 'G_loss':
            ylabel = 'Generator loss'
        else:
            ylabel = 'Discriminator loss'

        for i in values:
            fig = plt.figure(figsize=(12, 5))
            chart = fig.add_subplot()
            chart.set_xlabel("Epochs")
            chart.set_ylabel(ylabel)
            chart.plot(x, mapper[i], color='b')
            fig.savefig("{0}{1}_{2}_{3}.pdf".format(path, index, param, i))
            plt.close(fig)

    else:
        for i in values:
            fig = plt.figure(figsize=(12, 5))
            chart = fig.add_subplot()
            chart.set_xlabel("Epochs")
            chart.set_ylabel("Probability")
            sr = mapper[index[0]]
            hr = mapper[index[1]]
            chart.plot(x, sr[i], color='b', label="SR")
            chart.plot(x, hr[i], color='r', label="HR")
            chart.legend()
            fig.savefig("{0}{1}_{2}_{3}.pdf".format(path, "Train_Prob" if index == ['sr_prob', 'hr_prob'] else "Validate_Prob", param, i))
            plt.close(fig)
    # else:
    #     prob = mapper[index[0]]
    #     PROB = mapper[index[1]]
    #     idx = 0
    #     for i in values:
    #         fig = plt.figure(figsize=(12, 5))
    #         chart = fig.add_subplot()
    #         chart.set_xlabel("Epochs")
    #         chart.set_ylabel("{} Prob".format("SR" if index == ['sr_prob', 'SR_PROB'] else "HR"))
    #         chart.plot(x, prob[i], color='b', label='Training')
    #         chart.plot(x, PROB[i], color='r', label='Validate')
    #         chart.legend()
    #         fig.savefig("{0}{1}_{2}_{3}.pdf".format(path, "SR_PROB" if index == ['sr_prob', 'SR_PROB'] else "HR_PROB", param, i))
    #         plt.close(fig)

def plot(param):
    """
    :param param: one of ['first_kernel_size', 'BN', 'n_channels', 'n_res_blocks']
    :return:
    """
    if param == 'D_first_kernel_size':
        values = [3, 5, 7]
    elif param == 'n_blocks':
        values = ['5, 7', '7, 9', '5, 3', '3, 3', '3, 7']
    elif param == 'classifier':
        values = ['CNN', 'Dense']
    elif param == 'BN':
        values = ['with BN', 'without BN']
    else:
        print("Param not valid")
        return -1

    input_file = "logs_{}.txt".format(param)
    with open("../{}".format(input_file), "r") as fr:
        raw = fr.read()

    data = re.findall(r"EPOCH\W+(\d+)\W+D_loss\W+([\d\.e-]+)\W+sr_prob\W+([\d\.e-]+)\W+hr_prob\W+([\d\.e-]+)\W+G_loss\W+([\d\.e-]+)" \
                      "(?s:.*?)PSNR\W+([\d\.]+)\W+SSIM\W+([\d\.]+)\W+SR_PROB\W+([\d\.e-]+)\W+HR_PROB\W+([\d\.e-]+)", raw)
    dt = np.dtype([('EPOCH', 'int'), ('D_loss', 'float'), ('sr_prob', 'float'), ('hr_prob', 'float'), ('G_loss', 'float'),
                   ('PSNR', 'float'), ('SSIM', 'float'), ('SR_PROB', 'float'), ('HR_PROB', 'float')])
    data = np.array(data, dtype=dt)
    rows = len(data) // 50
    data.shape = (rows, 50)

    PSNR = {}
    SSIM = {}
    G_loss = {}
    D_loss = {}
    sr_prob = {}
    SR_PROB = {}
    hr_prob = {}
    HR_PROB = {}
    idx = 0
    for i in values:
        PSNR[i] = np.array([x["PSNR"] for x in data[idx]])
        SSIM[i] = np.array([x["SSIM"] for x in data[idx]])
        G_loss[i] = np.array([x["G_loss"] for x in data[idx]])
        D_loss[i] = np.array([x["D_loss"] for x in data[idx]])
        sr_prob[i] = np.array([x["sr_prob"] for x in data[idx]])
        SR_PROB[i] = np.array([x["SR_PROB"] for x in data[idx]])
        hr_prob[i] = np.array([x["hr_prob"] for x in data[idx]])
        HR_PROB[i] = np.array([x["HR_PROB"] for x in data[idx]])

        idx += 1

    indices = ['PSNR', 'SSIM', 'D_loss', 'G_loss', ['sr_prob', 'hr_prob'], ['SR_PROB', 'HR_PROB']]
    for index in indices:

        if index == 'PSNR': mapper = PSNR
        elif index == 'SSIM': mapper = SSIM
        elif index == 'G_loss': mapper = G_loss
        elif index == 'D_loss': mapper = D_loss
        else:
            if index == ['sr_prob', 'hr_prob']:
                mapper = {'sr_prob': sr_prob, 'hr_prob': hr_prob}
            else:
                mapper = {'SR_PROB': SR_PROB, 'HR_PROB': HR_PROB}
        plot_graph(index, param, values, mapper=mapper)

params = ['D_first_kernel_size', 'n_blocks', 'classifier', 'BN']
for param in params:
    plot(param)



