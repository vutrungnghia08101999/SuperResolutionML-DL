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
Test
===
```bash
python test.py --weights-file "/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/outputs/x3/epoch_19.pth" \
               --test-file "/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/test/BSDS100.h5" \
               --scale 3
```
