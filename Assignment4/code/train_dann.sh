export CUDA_VIDIBLE_DEVICES=0,1
python train_DANN.py --lr 1e-4 \
                     --epochs 100 \
                     --device "cuda:1" \
                     --model "DANN" \
                     --_lambda -1.0 \
                     --batch_size 128 \
