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
