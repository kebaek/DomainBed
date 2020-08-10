CUDA_VISIBLE_DEVICES=0,1,3 python -m domainbed.scripts.train --data_dir data --output_dir output/mcr/256_60_e0 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 0 --test_envs 0
CUDA_VISIBLE_DEVICES=0,1,3 python -m domainbed.scripts.train --data_dir data --output_dir output/mcr/256_60_e1 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 0 --test_envs 1
CUDA_VISIBLE_DEVICES=0,1,3 python -m domainbed.scripts.train --data_dir data --output_dir output/mcr/256_60_e2 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 0 --test_envs 2
CUDA_VISIBLE_DEVICES=0,1,3 python -m domainbed.scripts.train --data_dir data --output_dir output/mcr/256_60_e3 --algorithm MCR --dataset RotatedMNIST --hparams_seed 0 --test_envs 3
CUDA_VISIBLE_DEVICES=0,1,3 python -m domainbed.scripts.train --data_dir data --output_dir output/mcr/256_60_e4 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 0 --test_envs 4
CUDA_VISIBLE_DEVICES=0,1,3 python -m domainbed.scripts.train --data_dir data --output_dir output/mcr/256_60_e5 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 0 --test_envs 5

