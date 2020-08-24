#CUDA_VISIBLE_DEVICES=5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/pacs/erm/beta_tests/b0.001 --algorithm ERM  --dataset PACS --hparams_seed 0 --test_envs 0 --beta 0.001
#CUDA_VISIBLE_DEVICES=5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/pacs/erm/beta_tests/b0.0001 --algorithm ERM  --dataset PACS --hparams_seed 0 --test_envs 0 --beta 0.0001
CUDA_VISIBLE_DEVICES=5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/pacs/erm/beta_tests/b0.01 --algorithm ERM --dataset PACS --hparams_seed 0 --test_envs 0 --beta 0.01
CUDA_VISIBLE_DEVICES=5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/pacs/erm/beta_tests/b0 --algorithm ERM  --dataset PACS --hparams_seed 0 --test_envs 0
