CUDA_VISIBLE_DEVICES=5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/pacs/erm/default_e0 --algorithm ERM  --dataset PACS --hparams_seed 0 --test_envs 0
CUDA_VISIBLE_DEVICES=5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/pacs/erm/default_e1 --algorithm ERM  --dataset PACS --hparams_seed 0 --test_envs 1
CUDA_VISIBLE_DEVICES=5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/pacs/erm/default_e2 --algorithm ERM  --dataset PACS --hparams_seed 0 --test_envs 2
CUDA_VISIBLE_DEVICES=5,6,7 python -m domainbed.scripts.train --data_dir data --output_dir output/pacs/erm/default_e3 --algorithm ERM --dataset PACS --hparams_seed 0 --test_envs 3

