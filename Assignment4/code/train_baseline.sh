export CUDA_VIDIBLE_DEVICES=0,1
python train_baseline.py --lr 1e-5 \
                     --epochs 100 \
                     --device "cuda:1" \
                     --model "Baseline" \
                     --batch_size 128 \
