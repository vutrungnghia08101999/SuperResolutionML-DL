import re
import numpy as np
import matplotlib.pyplot as plt

with open("../logs.txt", "r") as fr:
    raw = fr.read()

data = re.findall(r"EPOCH\W+(\d+)\W+Loss\W+([\d\.e-]+)(?s:.*?)Val_loss\W+([\d\.e-]+)\W+PSNR\W+" \
                  "([\d\.e-]+)\W+SSIM\W+([\d\.e-]+)", raw)
dt = np.dtype([("EPOCH", "int"), ("Loss", "float"), ("Val_loss", "float"), ("PSNR", "float"), ("SSIM", "float")])
data = np.array(data, dtype=dt)

indices = ["Loss", "PSNR", "SSIM"]
loss = {"train": [item["Loss"] for item in data], "val": [item["Val_loss"] for item in data]}
PSNR = [item["PSNR"] for item in data]
SSIM = [item["SSIM"] for item in data]
dictionary = {"Loss": loss, "PSNR": PSNR, "SSIM": SSIM}
x = [item["EPOCH"] for item in data]
colors = ['b', 'r']
def plot(index, data):
    if index == "Loss":
        ys = [data["train"], data["val"]]
        labels = ["Train", "Validate"]
    else:
        ys = [data]
        labels = [index]
    length = len(ys)
    fig = plt.figure(figsize=(12,5))
    chart = fig.add_subplot()
    chart.set_xlabel("Epochs")
    chart.set_ylabel(index)
    for i in range(length):
        y = ys[i]
        label = labels[i]
        chart.plot(x, y, label=label, color=colors[i])
    if length != 1:
        chart.legend()
    fig.savefig("../{}.pdf".format(index))
    plt.close(fig)

for index in indices:
    plot(index, dictionary[index])
