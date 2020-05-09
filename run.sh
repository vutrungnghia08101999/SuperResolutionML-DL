# python train.py --kernel_size 3 \
#                 --train_file '/content/gdrive/My Drive/MOUNT/dataset/train-toy.pkl'\
#                 --valid_file '/content/gdrive/My Drive/MOUNT/dataset/valid-toy_4.pkl' \
#                 --models_dir '/content/gdrive/My Drive/MOUNT/models'

python train.py --kernel_size 3 \
                --train_file '/content/gdrive/My Drive/MOUNT/dataset/VOC-2012-train.pkl'\
                --valid_file '/content/gdrive/My Drive/MOUNT/dataset/VOC-2012-valid_4.pkl' \
                --models_dir '/content/gdrive/My Drive/MOUNT/models'

python train.py --kernel_size 5 \
                --train_file '/content/gdrive/My Drive/MOUNT/dataset/VOC-2012-train.pkl'\
                --valid_file '/content/gdrive/My Drive/MOUNT/dataset/VOC-2012-valid_4.pkl' \
                --models_dir '/content/gdrive/My Drive/MOUNT/models'

python train.py --kernel_size 7 \
                --train_file '/content/gdrive/My Drive/MOUNT/dataset/VOC-2012-train.pkl'\
                --valid_file '/content/gdrive/My Drive/MOUNT/dataset/VOC-2012-valid_4.pkl' \
                --models_dir '/content/gdrive/My Drive/MOUNT/models'

python train.py --kernel_size 9 \
                --train_file '/content/gdrive/My Drive/MOUNT/dataset/VOC-2012-train.pkl'\
                --valid_file '/content/gdrive/My Drive/MOUNT/dataset/VOC-2012-valid_4.pkl' \
                --models_dir '/content/gdrive/My Drive/MOUNT/models'
