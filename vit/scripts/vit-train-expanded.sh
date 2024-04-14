CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=12 torchrun --nproc-per-node=4 --master_port=2399 main.py --batch-size 256 --data-path /dataset/imagenet/ --output_dir /data/vit/train_expanded/ --resume "/data/vit/expanded/deit_small_patch16_224/expanded_model.pth" "$@"

