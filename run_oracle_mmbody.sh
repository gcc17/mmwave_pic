for RATIO in 0.45 0.475 0.5
do 
    CUDA_VISIBLE_DEVICES=3 python main_oracle.py --config configs/mmbody_oracle.yaml --select_ratio $RATIO --model_ckpt_path checkpoints/MMBody_PointTransformer/epoch80_mpjpe70.5222_pmpjpe53.7661.pth
done
