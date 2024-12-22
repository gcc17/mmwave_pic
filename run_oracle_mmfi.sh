for RATIO in 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5
do 
    CUDA_VISIBLE_DEVICES=2 python main_oracle.py --config configs/mmfi_oracle.yaml --select_ratio $RATIO --model_ckpt_path checkpoints/MMFi-PointTransformer-original-2024-12-19_11-40-13/epoch50_mpjpe82.1213_pmpjpe61.3635.pth
done
