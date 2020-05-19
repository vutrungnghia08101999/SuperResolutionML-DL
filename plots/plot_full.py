import re
import numpy as np
import matplotlib.pyplot as plt

with open("../logs.txt", "r") as fr:
    raw = fr.read()

# data = re.findall("EPOCH\W+(\d+)(?s:.*?)sr_prob\W+([\d\.e-]+)\W+hr_prob\W+([\d\.e-]+)(?s:.*?)" \
#                   "SR_PROB\W+([\d\.e-]+)\W+HR_PROB\W+([\d\.e-]+)", raw)
# dt = np.dtype([("EPOCH", "int"), ("sr_prob", "float"), ("hr_prob", "float"),
#                ("SR_PROB", "float"), ("HR_PROB", "float")])
data = re.findall("EPOCH\W+(\d+)\W+D_loss\W+([\d\.e-]+)\W+sr_prob\W+([\d\.e-]+)\W+"
                  "hr_prob\W+([\d\.e-]+)\W+G_loss\W+([\d\.e-]+)(?s:.*?)PSNR\W+([\d\.e-]+)"
                  "\W+SSIM\W+([\d\.e-]+)", raw)
dt = np.dtype([("EPOCH", "int"), ("D_loss", "float"), ("sr_prob", "float"), ("hr_prob", "float"),
               ("G_loss", "float"), ("PSNR", "float"), ("SSIM", "float")])
data = np.array(data, dtype=dt)

PSNR = [item["PSNR"] for item in data]
SSIM = [item["SSIM"] for item in data]
D_loss = [item["D_loss"] for item in data]
G_loss = [item["G_loss"] for item in data]
indices = ["PSNR", "SSIM", "D_loss", "G_loss"]
x = [item["EPOCH"] for item in data]
def draw(index, data):
    fig = plt.figure(figsize=(12,5))
    chart = fig.add_subplot()
    chart.set_xlabel("Epochs")
    if index in ["PSNR", "SSIM"]:
        ylabel = index
    else:
        if index == "D_loss":
            ylabel = "Discriminator loss"
        else: ylabel = "Generator loss"
    chart.set_ylabel(ylabel)
    chart.plot(x, data)
    fig.savefig("../{}.pdf".format(index))
    plt.close(fig)

dictionary = {}
dictionary["PSNR"] = PSNR
dictionary["SSIM"] = SSIM
dictionary["D_loss"] = D_loss
dictionary["G_loss"] = G_loss
for index in indices:
    draw(index, dictionary[index])

# x = [item["EPOCH"] for item in data]
# sr_prob = [item["sr_prob"] for item in data]
# hr_prob = [item["hr_prob"] for item in data]
# SR_PROB = [item["SR_PROB"] for item in data]
# HR_PROB = [item["HR_PROB"] for item in data]

# train_fig = plt.figure(figsize=(12, 5))
# train_chart = train_fig.add_subplot()
# train_chart.set_ylabel("Train probability")
# train_chart.set_xlabel("Epochs")
# train_chart.plot(x, sr_prob, color='b', label="SR")
# train_chart.plot(x, hr_prob, color='r', label="HR")
# train_chart.legend()
# train_fig.savefig("../train_prob.pdf")
# plt.close(train_fig)
#
# val_fig = plt.figure(figsize=(12,5))
# val_chart = val_fig.add_subplot()
# val_chart.set_xlabel("Epochs")
# val_chart.set_ylabel("Validate probability")
# val_chart.plot(x, SR_PROB, color='b', label="SR")
# val_chart.plot(x, HR_PROB, color='r', label="HR")
# val_chart.legend()
# val_fig.savefig("../validate_prob.pdf")
# plt.close(val_fig)




