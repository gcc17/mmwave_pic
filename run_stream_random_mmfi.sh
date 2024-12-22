# for SEED in 420 219 1223 306 305
# do
#     for RATIO in 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5
#     do 
#         CUDA_VISIBLE_DEVICES=2 python main_stream_random.py --config configs/mmfi_stream_random.yaml --random_ratio $RATIO --random_seed $SEED 
#     done

# done


for SEED in 420 219 1223 306 305
do
    CUDA_VISIBLE_DEVICES=2 python main_stream_random_v2.py --config configs/mmfi_stream_random.yaml --random_seed $SEED 
done
