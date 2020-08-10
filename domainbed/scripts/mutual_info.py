# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import torch
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Domain generalization')
	parser.add_argument('--data_dir', type=str)
	parser.add_argument('--dataset', type=str, default="RotatedMNIST")
	parser.add_argument('--algorithm', type=str, default="ERM")
	parser.add_argument('--hparams', type=str,
		help='JSON-serialized hparams dict')
	parser.add_argument('--hparams_seed', type=int, default=0,
		help='Seed for random hparams (0 means "default hparams")')
	parser.add_argument('--trial_seed', type=int, default=0,
		help='Trial number (used for seeding split_dataset and '
		'random_hparams).')
	parser.add_argument('--seed', type=int, default=0,
		help='Seed for everything else')
	parser.add_argument('--steps', type=int, default=None,
		help='Number of steps. Default is dataset-dependent.')
	parser.add_argument('--checkpoint_freq', type=int, default=None,
		help='Checkpoint every N steps. Default is dataset-dependent.')
	parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
	parser.add_argument('--file', type=str, default="G.pth.tar")
	parser.add_argument('--holdout_fraction', type=float, default=0.2)
	args = parser.parse_args()

	# If we ever want to implement checkpointing, just persist these values
	# every once in a while, and then load them from disk here.
	start_step = 0
	algorithm_dict = None

	print('Args:')
	for k, v in sorted(vars(args).items()):
		print('\t{}: {}'.format(k, v))

	if args.hparams_seed == 0:
		hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
	else:
		hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
			misc.seed_hash(args.hparams_seed, args.trial_seed))
	if args.hparams:
		hparams.update(json.loads(args.hparams))

	print('HParams:')
	for k, v in sorted(hparams.items()):
		print('\t{}: {}'.format(k, v))

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	if torch.cuda.is_available():
		device = "cuda"
	else:
		device = "cpu"

	if args.dataset in vars(datasets):
		dataset = vars(datasets)[args.dataset](args.data_dir,
			args.test_envs, hparams)
	else:
		raise NotImplementedError

	# Split each env into an 'in-split' and an 'out-split'. We'll train on
	# each in-split except the test envs, and evaluate on all splits.
	in_splits = []
	out_splits = []
	for env_i, env in enumerate(dataset):
		out, in_ = misc.split_dataset(env,
			int(len(env)*args.holdout_fraction),
			misc.seed_hash(args.trial_seed, env_i))
		if hparams['class_balanced']:
			in_weights = misc.make_weights_for_balanced_classes(in_)
			out_weights = misc.make_weights_for_balanced_classes(out)
		else:
			in_weights, out_weights = None, None
		in_splits.append((in_, in_weights))
		out_splits.append((out, out_weights))

	train_loaders = [FastDataLoader(
		dataset=env,
		weights=env_weights,
		batch_size=hparams['batch_size'],
		num_workers=dataset.N_WORKERS,
		length=FastDataLoader.INFINITE)
		for i, (env, env_weights) in enumerate(in_splits)
		if i not in args.test_envs]

	eval_loaders = [FastDataLoader(
		dataset=env,
		weights=None,
		batch_size=64,
		num_workers=dataset.N_WORKERS,
		length=FastDataLoader.EPOCH)
		for env, _ in (in_splits + out_splits)]
	eval_weights = [None for _, weights in (in_splits + out_splits)]
	eval_loader_names = ['env{}_in'.format(i)
		for i in range(len(in_splits))]
	eval_loader_names += ['env{}_out'.format(i)
		for i in range(len(out_splits))]

	algorithm_class = algorithms.get_algorithm_class(args.algorithm)
	algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
		len(dataset) - len(args.test_envs), hparams)

	algorithm.load_state_dict(torch.load(args.file))
	algorithm = algorithm.to(device)
    algorithm.eval()
    all_data = chain(*train_loaders)
    algorithm.update(all_data, components=True)

	train_minibatches_iterator = zip(*train_loaders)
    minibatches_device = [(x.to(device), y.to(device))
        for x,y in next(train_minibatches_iterator)]

    print('SVD Accuracy')
    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    for name, loader, weights in evals:
        acc = misc.accuracy(algorithm, loader, weights, device)
        results[name+'_acc'] = acc
    results_keys = sorted(results.keys())
    misc.print_row(results_keys, colwidth=12)
    misc.print_row([results[key] for key in results_keys], colwidth=12)

    print('SVD')
