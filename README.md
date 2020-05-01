Download dataset: https://drive.google.com/drive/folders/116ZLlLSrS2OLrsczYRv5nDpV9lL2cE0w?usp=sharing

Create train dataset
===
```bash
python prepare.py --images-dir '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/train/t91' \
                  --output-path '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/train/t91.h5' \
                  --scale 3
```

Create valid dataset
===
```bash
python prepare.py --images-dir '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/valid/Set14' \
                  --output-path '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/valid/Set14.h5' \
                  --scale 3 \
                  --eval
```
Create test dataset
===
```bash
python prepare.py --images-dir '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/test/BSDS100' \
                  --output-path '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/test/BSDS100.h5' \
                  --scale 3 \
                  --eval
```
Train and Evaluate
===
```bash
python train.py --train-file "/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/train/t91.h5" \
                --eval-file "/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/valid/Set14.h5" \
                --outputs-dir "/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/outputs" \
                --scale 3 \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 20 \
                --num-workers 8 \
                --seed 123
```

Inference
===
```bash
python test.py --weights-file "/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/ESPCN/pretrained-models/espcn_x3.pth" \
               --image-file "/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/ESPCN/inference/baboon.bmp" \
               --scale 3
```
