import re
import numpy as np
import matplotlib.pyplot as plt

with open("../logs.txt", "r") as fr:
    raw = fr.read()

data = re.findall("EPOCH\W+(\d+)(?s:.*?)sr_prob\W+([\d\.e-]+)\W+hr_prob\W+([\d\.e-]+)(?s:.*?)" \
                  "SR_PROB\W+([\d\.e-]+)\W+HR_PROB\W+([\d\.e-]+)", raw)
dt = np.dtype([("EPOCH", "int"), ("sr_prob", "float"), ("hr_prob", "float"),
               ("SR_PROB", "float"), ("HR_PROB", "float")])
data = np.array(data, dtype=dt)

x = [item["EPOCH"] for item in data]
sr_prob = [item["sr_prob"] for item in data]
hr_prob = [item["hr_prob"] for item in data]
SR_PROB = [item["SR_PROB"] for item in data]
HR_PROB = [item["HR_PROB"] for item in data]

train_fig = plt.figure(figsize=(12, 5))
train_chart = train_fig.add_subplot()
train_chart.set_ylabel("Train probability")
train_chart.set_xlabel("Epochs")
train_chart.plot(x, sr_prob, color='b', label="SR")
train_chart.plot(x, hr_prob, color='r', label="HR")
train_chart.legend()
train_fig.savefig("../train_prob.pdf")
plt.close(train_fig)

val_fig = plt.figure(figsize=(12,5))
val_chart = val_fig.add_subplot()
val_chart.set_xlabel("Epochs")
val_chart.set_ylabel("Validate probability")
val_chart.plot(x, SR_PROB, color='b', label="SR")
val_chart.plot(x, HR_PROB, color='r', label="HR")
val_chart.legend()
val_fig.savefig("../validate_prob.pdf")
plt.close(val_fig)




