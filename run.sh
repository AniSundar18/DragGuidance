NAME='data/lion.jpg'
#CUDA_LAUNCH_BLOCKING=1, CUDA_VISIBLE_DEVICES=3 python -u main.py --sd_version '1.4' --inversion_prompt "An image of a lion in a grassland" --data_path $NAME --condition_type 'image'

CUDA_LAUNCH_BLOCKING=1, CUDA_VISIBLE_DEVICES=3 python -u main.py --sd_version 'variations'  --data_path $NAME --condition_type 'image'  
