CUDA_VISIBLE_DEVICES=4,5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/svdsvm/svm_0 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 0 --test_envs 0
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/svdsvm/svm_1 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 0 --test_envs 1
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/svdsvm/svm_2 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 0 --test_envs 2
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/svdsvm/svm_3 --algorithm MCR --dataset RotatedMNIST --hparams_seed 0 --test_envs 3
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/svdsvm/svm_4 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 0 --test_envs 4
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/svdsvm/svm_5 --algorithm MCR  --dataset RotatedMNIST --hparams_seed 0 --test_envs 5
