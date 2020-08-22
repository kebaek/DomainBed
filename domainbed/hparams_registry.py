# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np


def _hparams(algorithm, dataset, random_state):
	"""
	Global registry of hyperparams. Each entry is a (default, random) tuple.
	New algorithms / networks / etc. should add entries here.
	"""
	RESNET_DATASETS = ['VLCS', 'PACS', 'OfficeHome', 'TerraIncognita',
		'DomainNet']

	hparams = {}

	fd = int(random_state.uniform(2, 8))
	hparams['fd']=(2048,10)
	hparams['beta'] = (0,0.5)
	if dataset in RESNET_DATASETS:
		hparams['lr'] = (5e-5, 10**random_state.uniform(-5, -3.5))
		hparams['batch_size'] = (64, int(2**random_state.uniform(3, 5.5)))
	else:
		hparams['lr'] = (1e-3, 10**random_state.uniform(-4.5, -2.5))
		#hparams['batch_size'] = (128, int(2**random_state.uniform(3, 9)))
		#hparams['batch_size'] = (512, fd * int(random_state.uniform(3, 9)))
		hparams['batch_size']=(256,256)
	if dataset in ['ColoredMNIST', 'RotatedMNIST']:
		hparams['weight_decay'] = (0., 0.)
	else:
		hparams['weight_decay'] = (0., 10**random_state.uniform(-6, -2))

	hparams['class_balanced'] = (True, False)
	if algorithm in ['DANN', 'CDANN']:

		if dataset in RESNET_DATASETS:
			hparams['lr_g'] = (5e-5, 10**random_state.uniform(-5, -3.5))
			hparams['lr_d'] = (5e-5, 10**random_state.uniform(-5, -3.5))
		else:
			hparams['lr_g'] = (1e-3, 10**random_state.uniform(-4.5, -2.5))
			hparams['lr_d'] = (1e-3, 10**random_state.uniform(-4.5, -2.5))

		if dataset in ['ColoredMNIST', 'RotatedMNIST']:
			hparams['weight_decay_g'] = (0., 0.)
		else:
			hparams['weight_decay_g'] = (0., 10**random_state.uniform(-6, -2))

		hparams['lambda'] = (1.0, 10**random_state.uniform(-2, 2))
		hparams['weight_decay_d'] = (0., 10**random_state.uniform(-6, -2))
		hparams['d_steps_per_g_step'] = (1, int(2**random_state.uniform(0, 3)))
		hparams['grad_penalty'] = (0., 10**random_state.uniform(-2, 1))
		hparams['beta1'] = (0.5, random_state.choice([0., 0.5]))

	hparams['resnet_dropout'] = (0., float(random_state.choice([0., 0.1, 0.5])))

	# TODO clean this up
	hparams.update({a:(b,c) for a,b,c in [
		# IRM
		('irm_lambda', 1e2, 10**random_state.uniform(-1, 5)),
		('irm_penalty_anneal_iters', 500, int(10**random_state.uniform(0, 4))),
		# Mixup
		('mixup_alpha', 0.2, 10**random_state.uniform(-1, -1)),
		# GroupDRO
		('groupdro_eta', 1e-2, 10**random_state.uniform(-3, -1)),
		# MMD
		('mmd_gamma', 1., 10**random_state.uniform(-1, 1)),
		# MLP
		('mlp_width', 256, int(2**random_state.uniform(6, 10))),
		('mlp_depth', 3, int(random_state.choice([3,4,5])) ),
		('mlp_dropout', 0., float(random_state.choice([0., 0.1, 0.5])) ),
		# MLDG
		('mldg_beta', 1., 10**random_state.uniform(-1, 1))
	]})
	return hparams

def default_hparams(algorithm, dataset):
	dummy_random_state = np.random.RandomState(0)
	return {a: b for a,(b,c) in
		_hparams(algorithm, dataset, dummy_random_state).items()}

def random_hparams(algorithm, dataset, seed):
	random_state = np.random.RandomState(seed)
	return {a: c for a,(b,c) in _hparams(algorithm, dataset, random_state).items()}
