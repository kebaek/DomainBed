CUDA_VISIBLE_DEVICES=0,1 python -m domainbed.scripts.train --data_dir data --output_dir output/256_60_h0 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 0 --test_envs 0
CUDA_VISIBLE_DEVICES=0,1 python -m domainbed.scripts.train --data_dir data --output_dir output/256_60_h1 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 1 --test_envs 0 
CUDA_VISIBLE_DEVICES=0,1 python -m domainbed.scripts.train --data_dir data --output_dir output/256_60_h2 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 2 --test_envs 0
CUDA_VISIBLE_DEVICES=0,1 python -m domainbed.scripts.train --data_dir data --output_dir output/256_60_h3 --algorithm MCR --dataset RotatedMNIST --hparams_seed 3 --test_envs 0
CUDA_VISIBLE_DEVICES=0,1 python -m domainbed.scripts.train --data_dir data --output_dir output/256_60_h4 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 4 --test_envs 0
CUDA_VISIBLE_DEVICES=0,1 python -m domainbed.scripts.train --data_dir data --output_dir output/256_60_h5 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 5 --test_envs 0

