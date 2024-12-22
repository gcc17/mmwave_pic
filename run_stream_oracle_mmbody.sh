for RATIO in 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5
do 
    CUDA_VISIBLE_DEVICES=2 python main_stream_oracle.py --config configs/mmbody_stream_oracle.yaml --select_ratio $RATIO --model_ckpt_path checkpoints/MMBody_PointTransformer/epoch80_mpjpe70.5222_pmpjpe53.7661.pth
done
