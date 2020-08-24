CUDA_VISIBLE_DEVICES=2,3 python -m domainbed.scripts.train --data_dir data --output_dir output/mnist/svm_0 --algorithm Union  --dataset RotatedMNIST --hparams_seed 0 --test_envs 0
CUDA_VISIBLE_DEVICES=2,3 python -m domainbed.scripts.train --data_dir data --output_dir output/mnist/svm_1 --algorithm Union  --dataset RotatedMNIST --hparams_seed 0 --test_envs 1
CUDA_VISIBLE_DEVICES=2,3 python -m domainbed.scripts.train --data_dir data --output_dir output/mnist/svm_2 --algorithm Union  --dataset RotatedMNIST --hparams_seed 0 --test_envs 2
CUDA_VISIBLE_DEVICES=2,3 python -m domainbed.scripts.train --data_dir data --output_dir output/mnist/svm_3 --algorithm Union --dataset RotatedMNIST --hparams_seed 0 --test_envs 3
CUDA_VISIBLE_DEVICES=2,3 python -m domainbed.scripts.train --data_dir data --output_dir output/mnist/svm_4 --algorithm Union  --dataset RotatedMNIST --hparams_seed 0 --test_envs 4
CUDA_VISIBLE_DEVICES=2,3 python -m domainbed.scripts.train --data_dir data --output_dir output/mnist/svm_5 --algorithm Union  --dataset RotatedMNIST --hparams_seed 0 --test_envs 5
