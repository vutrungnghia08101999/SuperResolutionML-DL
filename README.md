SRResNet
===

Train
```bash
python train.py --train_file '/content/gdrive/My Drive/MOUNT/dataset/VOC-2012-train.pkl'\
                --valid_file '/content/gdrive/My Drive/MOUNT/dataset/VOC-2012-valid_4.pkl' \
                --models_dir '/content/gdrive/My Drive/MOUNT/models'
```
Test
```bash
python test.py --model_path '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/models/checkpoint_20-05-12_17-31-07_100.pth' \
               --test_folder '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/BSDS100/SRF_4' \
               --output '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/testing-world-champion/SRResNet/X4'
```
Inference
```bash
python inference.py --model_path '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/models/checkpoint_20-05-12_17-31-07_100.pth' \
                    --output './' \
                    --image_name 'tmp.png'
```
