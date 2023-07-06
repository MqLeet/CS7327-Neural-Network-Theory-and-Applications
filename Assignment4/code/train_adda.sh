export CUDA_VIDIBLE_DEVICES=0,1
python train_ADDA.py --lr 1e-5 \
                     --epochs_pretrain 100 \
                     --epochs_adapt 200 \
                     --device "cuda:1" \
                     --model "ADDA" \
                     --batch_size 128 \
