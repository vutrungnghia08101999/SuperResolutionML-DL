Data processing
===
Download dataset: https://drive.google.com/drive/folders/116ZLlLSrS2OLrsczYRv5nDpV9lL2cE0w?usp=sharing

Create train dataset
```bash
python prepare.py --images-dir '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/train/t91' \
                  --scale 2
```

Create valid dataset
```bash
python prepare.py --images-dir '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/valid/Set14' \
                  --eval \
                  --scale 2
```
Create test dataset
```bash
python prepare.py --images-dir '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/test/BSDS100' \
                  --eval \
                  --scale 2
```
Train and Evaluate
===
```bash
python train.py --train-dir "/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/train/t91" \
                --eval-dir "/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/valid/Set14" \
                --outputs-dir "/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/models" \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 200 \
                --num-workers 8 \
                --seed 123 \
                --scale 2 --model 'ESPCN' --loss 'mse'
```
