CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/128_0 --algorithm MCR --dataset RotatedMNIST --hparams_seed 1 --test_envs 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/128_1 --algorithm MCR --dataset RotatedMNIST --hparams_seed 1 --test_envs 1 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/128_2 --algorithm MCR --dataset RotatedMNIST --hparams_seed 1 --test_envs 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/128_3 --algorithm MCR --dataset RotatedMNIST --hparams_seed 1 --test_envs 3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/128_4 --algorithm MCR --dataset RotatedMNIST --hparams_seed 1 --test_envs 4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/128_5 --algorithm MCR --dataset RotatedMNIST --hparams_seed 1 --test_envs 5

