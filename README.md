# SRGAN
### Train
```
python train.py

optional arguments:
--crop_size                   training images crop size [default value is 88]
--upscale_factor              super resolution upscale factor [default value is 4](choices:[2, 4, 8])
--num_epochs                  train epoch number [default value is 100]
```

### Test Benchmark Datasets
```
python test_benchmark.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution images are on `benchmark_results` directory.

### Test Single Image
```
python inference.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--test_mode                   using GPU or CPU [default value is 'GPU'](choices:['GPU', 'CPU'])
--image_name                  test low resolution image name
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
