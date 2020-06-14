Data Preprocessing
===
Download data and pretrained models at: https://husteduvn-my.sharepoint.com/:f:/g/personal/nghia_vt173284_sis_hust_edu_vn/EozFDBAS77dJmsbGvRvDswIB5O0P6ucF5OcU9U10jN8NvQ?e=XHVAb9

Generate train and test dataset:
- Save data at "dataset" folder
- Save models at "pretrained-models" folder
- Create "logs" folder
```bash
python generate_data.py
```
- Only generate scale 4 dataset for test dataset
- Simply compress images in train folder to pkl files to load faster in colab

Training
===
Output includes:
- logs file at logs folder
- model checkpoint at models folder
- images at experiments folder

Train ESPCN
```
python train_ESPCN.py --num_epochs 10
python train_ESPCN.py --weights pretrained-models/ESPCN.py --num_epochs 10
```
Train SRResNet
```
python train_SRResNet.py --num_epochs 1
python train_SRResNet.py --weights pretrained-models/SRResNet.py --num_epochs 1
```
Train SRGAN
```
python train_SRGAN.py --num_epochs 1
python train_SRGAN.py --weights pretrained-models/SRResNet.py --num_epochs 1
```
Testing
===
Output includes:
- logs file at logs folder
- image at experiment folder

Test ESPCN
```
python test --model ESPCN --weights pretrained-models/ESPCN.pth
```
Test SRResNet
```
python test --model SRResNet --weights pretrained-models/SRResNet.pth
```
Test SRGAN
```
python test --model SRGAN --weights pretrained-models/SRGAN.pth
```
