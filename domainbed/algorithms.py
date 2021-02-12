# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.mcr import MaximalCodingRateReduction, MutualInformation
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from itertools import combinations

ALGORITHMS = [
	'ERM',
	'DANN',
	'CHDANN',
	'CDANN',
	'IRM',
	'Mixup',
	'GroupDRO',
	'MLDG',
	'MMD',
	'CORAL'
	'MCR',
    'Union'
	'MCRI'
]

if torch.cuda.is_available():
	device = "cuda"
else:
	device = "cpu"

def get_algorithm_class(algorithm_name):
	"""Return the algorithm class with the given name."""
	if algorithm_name not in globals():
		raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
	return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
	"""
	A subclass of Algorithm implements a domain generalization algorithm.
	Subclasses should implement the following:
	- update()
	- predict()
	"""
	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(Algorithm, self).__init__()
		self.hparams = hparams
		self.num_domains = num_domains
		self.num_classes = num_classes

	def update(self, minibatches):
		"""
		Perform one update step, given a list of (x, y) tuples for all
		environments.
		"""
		raise NotImplementedError

	def predict(self, x):
		raise NotImplementedError


class ERM(Algorithm):
	"""
	Cross Entropy + Conditional Mutual Info (ERM + MCR)
	"""

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(ERM, self).__init__(input_shape, num_classes, num_domains,
								  hparams)
		self.featurizer = torch.nn.DataParallel(networks.Featurizer(input_shape, self.hparams))
		self.classifier = nn.Linear(hparams['fd'], num_classes)
		self.network = nn.Sequential(self.featurizer, self.classifier)
		self.networks = [self.featurizer]
		self.optimizer = torch.optim.Adam(
			self.network.parameters(),
			lr=self.hparams["lr"],
			weight_decay=self.hparams['weight_decay']
		)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, hparams['decay'], gamma=hparams['beta'], last_epoch=-1)
		self.cmi = MaximalCodingRateReduction(gam1=1, gam2=1, eps=0.5).to(device)
		self.components = {}
		self.singular_values = {}

	def update(self, minibatches, components=False):
		if components:
			p = []
			all_y = []
			for x,y in minibatches:
				p.append(self.featurizer(x.cuda()).cpu().detach())
				all_y.append(y)
			p, all_y = torch.cat(p), torch.cat(all_y)
			self.svd(p, all_y)
			return None
		else:
			all_x = torch.cat([x for x,y in minibatches])
			all_y = torch.cat([y for x,y in minibatches])
			all_z = self.featurizer(all_x).cuda()
			ce = F.cross_entropy(self.classifier(all_z), all_y)
			loss = ce
			mi = torch.tensor(0)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			self.scheduler.step()

			return {'loss': loss.item(), 'ce': ce.item(), 'mi': mi.item()}

	def svd(self, x, y):
		sorted_data = [[] for _ in range(self.num_classes)]
		for i, lbl in enumerate(y):
			sorted_data[lbl].append(x[i])
		sorted_data = [torch.stack(class_data).cpu() for class_data in sorted_data]

		for j in range(self.num_classes):
			u,s,vt = torch.svd(sorted_data[j])
			self.components[j] = vt.t()[:self.hparams['n_comp']]
			self.singular_values[j] = s[:self.hparams['n_comp']]

	def predict(self, x):
		return self.network(x)

class MCR(Algorithm):
	"""
	Maximal Coding Rate Reduction (MCR)
	"""

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(MCR, self).__init__(input_shape, num_classes, num_domains,
								  hparams)
		self.featurizer = torch.nn.DataParallel(networks.Featurizer(input_shape, self.hparams))
		self.network = self.featurizer
		self.networks = [self.featurizer]
		self.optimizer = torch.optim.Adam(
			self.featurizer.parameters(),
			lr=self.hparams["lr"],
			weight_decay=self.hparams['weight_decay']
		)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, hparams['decay'], gamma=hparams['beta'], last_epoch=-1)
		self.criterion = MaximalCodingRateReduction(gam1=hparams['gam1'], gam2=hparams['gam2'], eps=hparams['eps']).to(device)
		self.components = {}
		self.singular_values = {}

	def update(self, minibatches, components=False):
		if components:
			p = []
			all_y = []
			for x,y in minibatches:
				p.append(self.featurizer(x.cuda()).cpu().detach())
				all_y.append(y)
			p, all_y = torch.cat(p), torch.cat(all_y)
			self.svd(p, all_y)
			return None
		else:
			all_z = self.featurizer(torch.cat([x for x,y in minibatches]))
			if self.hparams['norm']==2:
				all_z = len(all_z)*all_z/torch.norm(all_z,p='fro')
			all_y = torch.cat([y for x,y in minibatches])
			mcr, loss_empi, loss_theo = self.criterion(all_z, all_y, self.num_classes)
			loss = mcr
			mi = torch.tensor(0, dtype=torch.float32).cuda()

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			self.scheduler.step()
			return {'loss': loss.item(), 'mcr': mcr.item(), 'mi':mi.item()}

	def svd(self, x, y):
		sorted_data = [[] for _ in range(self.num_classes)]
		for i, lbl in enumerate(y):
			sorted_data[lbl].append(x[i])
		sorted_data = [torch.stack(class_data).cpu() for class_data in sorted_data]

		for j in range(self.num_classes):
			u,s,vt = torch.svd(sorted_data[j])
			self.components[j] = vt.t()[:self.hparams['n_comp']]
			self.singular_values[j] = s[:self.hparams['n_comp']]
		return self.singular_values

	def svm(self,x,y):
		self.components = LinearSVC(verbose=0, random_state=10)
		self.components.fit(x.cpu().detach().numpy(),y.cpu().detach().numpy())

	def predict(self, x, weighted=True):
		x = self.featurizer(x)
		scores_svd = []
		for j in range(self.num_classes):
			svd_j = torch.matmul(torch.matmul(torch.pow(F.normalize(self.singular_values[j], dim=0),2)*self.components[j].t(),self.components[j]).to(device),x.t().to(device))
			score_svd_j = torch.norm(svd_j, dim=0)
			scores_svd.append(score_svd_j)
			p = torch.argmax(torch.stack(scores_svd), dim=0)
		return F.one_hot(p, self.num_classes)

class MCRI(Algorithm):
	"""
	Maximal Coding Rate Reduction Individual Domain (MCRI)
	"""

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(MCRI, self).__init__(input_shape, num_classes, num_domains,
								  hparams)
		self.featurizer = torch.nn.DataParallel(networks.Featurizer(input_shape, self.hparams))
		self.network = self.featurizer
		self.networks = [self.featurizer]
		self.optimizer = torch.optim.Adam(
			self.featurizer.parameters(),
			lr=self.hparams["lr"],
			weight_decay=self.hparams['weight_decay']
		)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, hparams['decay'], gamma=hparams['beta'], last_epoch=-1)
		self.criterion = MaximalCodingRateReduction(gam1=1, gam2=1, eps=0.5).to(device)
		self.components = {}
		self.singular_values = {}

	def update(self, minibatches, components=False):
		if components:
			p = []
			all_y = []
			for x,y in minibatches:
				p.append(self.featurizer(x.cuda()).cpu().detach())
				all_y.append(y)
			p, all_y = torch.cat(p), torch.cat(all_y)
			self.svd(p, all_y)
			return None
		else:
			loss = torch.tensor(0, dtype=torch.float32).cuda()
			mi = torch.tensor(0, dtype=torch.float32).cuda()
			if self.hparams['norm']==2:
				all_z = []
				for x,y in minibatches:
					z = self.featurizer(x)
					z = len(z)*z/torch.norm(z,p='fro')
					all_z.append(z)
				all_z = torch.cat(all_z)
			else:
				all_z = [self.featurizer(torch.cat([x for x,y in minibatches]))]
			all_y = []
			i = 0
			for x,y in minibatches:
				z = all_z[i:i+len(y)]
				all_y += [y]
				mcr, loss_empi, loss_theo = self.criterion(z, y, self.num_classes)
				mi += mcr
				i += len(y)
			all_y = torch.cat(all_y)
			mcr, loss_empi, loss_theo = self.criterion(all_z, all_y, self.num_classes)
			loss = mi + mcr

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			self.scheduler.step()

			return {'loss': loss.item(), 'mcr': mcr.item(), 'mi':mi.item()}


	def svd(self, x, y):
		sorted_data = [[] for _ in range(self.num_classes)]
		for i, lbl in enumerate(y):
			sorted_data[lbl].append(x[i])
		sorted_data = [torch.stack(class_data).cpu() for class_data in sorted_data]

		for j in range(self.num_classes):
			u,s,vt = torch.svd(sorted_data[j])
			self.components[j] = vt.t()[:self.hparams['n_comp']]
			self.singular_values[j] = s[:self.hparams['n_comp']]
		return self.singular_values

	def svm(self,x,y):
		self.components = LinearSVC(verbose=0, random_state=10)
		self.components.fit(x.cpu().detach().numpy(),y.cpu().detach().numpy())

	def predict(self, x, weighted=True):
		x = self.featurizer(x)
		scores_svd = []
		for j in range(self.num_classes):
			svd_j = torch.matmul(torch.matmul(F.normalize(self.singular_values[j], dim=0)*self.components[j].t(),self.components[j]).to(device),x.t().to(device))
			score_svd_j = torch.norm(svd_j, dim=0)
			scores_svd.append(score_svd_j)
			p = torch.argmax(torch.stack(scores_svd), dim=0)
		return F.one_hot(p, self.num_classes)


class AbstractDANN(Algorithm):
	"""Domain-Adversarial Neural Networks (abstract class)"""

	def __init__(self, input_shape, num_classes, num_domains,
				 hparams, conditional, class_balance, hot=False, wasserstein=False):

		super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
								  hparams)

		self.register_buffer('update_count', torch.tensor([0]))
		self.num_classes = num_classes
		self.conditional = conditional
		self.hot = hot
		self.class_balance = class_balance
		print(num_domains)
		# Algorithms
		self.featurizer = torch.nn.DataParallel(networks.Featurizer(input_shape, self.hparams))
		self.classifier = nn.Linear(hparams['fd'], num_classes)
		if self.conditional:
			if self.hot:
				self.discriminator = torch.nn.DataParallel(networks.MLP(hparams['fd']+num_classes,
					num_domains, self.hparams))
			else:
				self.discriminator = torch.nn.DataParallel(networks.MLP(hparams['fd'],
					num_domains, self.hparams))
		else:
			self.discriminator = networks.MLP(hparams['fd'],
			num_domains, self.hparams)
		self.class_embeddings = nn.Embedding(num_classes,
			hparams['fd'])

		# Optimizers
		self.disc_opt = torch.optim.Adam(
			(list(self.discriminator.parameters()) +
				list(self.class_embeddings.parameters())),
			lr=self.hparams["lr_d"],
			weight_decay=self.hparams['weight_decay_d'],
			betas=(self.hparams['beta1'], 0.9))

		self.gen_opt = torch.optim.Adam(
			(list(self.featurizer.parameters()) +
				list(self.classifier.parameters())),
			lr=self.hparams["lr_g"],
			weight_decay=self.hparams['weight_decay_g'],
			betas=(self.hparams['beta1'], 0.9))

		self.disc_scheduler = torch.optim.lr_scheduler.StepLR(self.disc_opt, hparams['decay'], gamma=hparams['beta'], last_epoch=-1)
		self.gen_scheduler = torch.optim.lr_scheduler.StepLR(self.gen_opt, hparams['decay'], gamma=hparams['beta'], last_epoch=-1)

	def update(self, minibatches):
		self.update_count += 1
		all_x = torch.cat([x for x, y in minibatches])
		all_y = torch.cat([y for x, y in minibatches])
		all_z = self.featurizer(all_x)
		if self.conditional:
			if self.hot:
				disc_input = torch.cat((all_z,F.one_hot(all_y, self.num_classes).float()),1)
			else:
				disc_input = all_z + self.class_embeddings(all_y)
		else:
			disc_input = all_z
		disc_out = self.discriminator(disc_input)
		disc_labels = torch.cat([
			torch.full((x.shape[0], ), i, dtype=torch.int64, device='cuda')
			for i, (x, y) in enumerate(minibatches)
		])

		if wasserstein:
			E = [torch.mean(disc_out[disc_labels==i],dim=0) for i in range(len(minibatches))]
			disc_loss = 0
			pair_list = list(combinations(range(len(E)), 2))
			for i,j in pair_list:
				disc_loss -= torch.sum(torch.abs(E[i]-E[j]))
		else:
			if self.class_balance:
				y_counts = F.one_hot(all_y).sum(dim=0)
				weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
				disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
				disc_loss = (weights * disc_loss).sum()
			else:
				disc_loss = F.cross_entropy(disc_out, disc_labels)

		disc_softmax = F.softmax(disc_out, dim=1)
		input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
			[disc_input], create_graph=True)[0]
		grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
		disc_loss += self.hparams['grad_penalty'] * grad_penalty

		d_steps_per_g = self.hparams['d_steps_per_g_step']
		if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

			self.disc_opt.zero_grad()
			disc_loss.backward()
			self.disc_opt.step()
			self.disc_scheduler.step()
			return {'disc_loss': disc_loss.item()}
		else:
			all_preds = self.classifier(all_z)
			classifier_loss = F.cross_entropy(all_preds, all_y)
			gen_loss = (classifier_loss +
						(self.hparams['lambda'] * -disc_loss))
			self.disc_opt.zero_grad()
			self.gen_opt.zero_grad()
			gen_loss.backward()
			self.gen_opt.step()
			self.gen_scheduler.step()
			return {'gen_loss': gen_loss.item()}

	def predict(self, x):
		return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
	"""Unconditional DANN"""
	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(DANN, self).__init__(input_shape, num_classes, num_domains,
			hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
	"""Conditional Embedding DANN"""
	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(CDANN, self).__init__(input_shape, num_classes, num_domains,
			hparams, conditional=True, class_balance=True)

class CHDANN(AbstractDANN):
	"""Conditional One hot DANN"""
	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(CHDANN, self).__init__(input_shape, num_classes, num_domains,
			hparams, conditional=True, class_balance=True, hot=True)

class WDANN(AbstractDANN):
	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(WDANN, self).__init__(input_shape, num_classes, num_domains,
			hparams, conditional=True, class_balance=True, hot=True, wasserstein=True)

class IRM(ERM):
	"""Invariant Risk Minimization"""

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(IRM, self).__init__(input_shape, num_classes, num_domains,
								  hparams)
		self.register_buffer('update_count', torch.tensor([0]))

	@staticmethod
	def _irm_penalty(logits, y):
		scale = torch.tensor(1.).cuda().requires_grad_()
		loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
		loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
		grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
		grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
		result = torch.sum(grad_1 * grad_2)
		return result

	def update(self, minibatches):
		penalty_weight = (self.hparams['irm_lambda'] if self.update_count
						  >= self.hparams['irm_penalty_anneal_iters'] else
						  1.0)
		nll = 0.
		penalty = 0.

		all_x = torch.cat([x for x,y in minibatches])
		all_logits = self.network(all_x)
		all_logits_idx = 0
		for i, (x, y) in enumerate(minibatches):
			logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
			all_logits_idx += x.shape[0]
			nll += F.cross_entropy(logits, y)
			penalty += self._irm_penalty(logits, y)
		nll /= len(minibatches)
		penalty /= len(minibatches)
		loss = nll + (penalty_weight * penalty)

		if self.update_count == self.hparams['irm_penalty_anneal_iters']:
			# Reset Adam, because it doesn't like the sharp jump in gradient
			# magnitudes that happens at this step.
			self.optimizer = torch.optim.Adam(
				self.network.parameters(),
				lr=self.hparams["lr"],
				weight_decay=self.hparams['weight_decay'])

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.update_count += 1
		return {'loss': loss.item(), 'nll': nll.item(),
			'penalty': penalty.item()}


class Mixup(ERM):
	"""
	Mixup of minibatches from different domains
	https://arxiv.org/pdf/2001.00677.pdf
	https://arxiv.org/pdf/1912.01805.pdf
	"""
	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(Mixup, self).__init__(input_shape, num_classes, num_domains,
									hparams)

	def update(self, minibatches):
		objective = 0

		for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
			lam = np.random.beta(self.hparams["mixup_alpha"],
								 self.hparams["mixup_alpha"])

			x = lam * xi + (1 - lam) * xj
			predictions = self.predict(x)

			objective += lam * F.cross_entropy(predictions, yi)
			objective += (1 - lam) * F.cross_entropy(predictions, yj)

		objective /= len(minibatches)

		self.optimizer.zero_grad()
		objective.backward()
		self.optimizer.step()

		return {'loss': objective.item()}


class GroupDRO(ERM):
	"""
	Robust ERM minimizes the error at the worst minibatch
	Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
	"""
	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
										hparams)
		self.register_buffer("q", torch.Tensor())

	def update(self, minibatches):
		device = "cuda" if minibatches[0][0].is_cuda else "cpu"

		if not len(self.q):
			self.q = torch.ones(len(minibatches)).to(device)

		losses = torch.zeros(len(minibatches)).to(device)

		for m in range(len(minibatches)):
			x, y = minibatches[m]
			losses[m] = F.cross_entropy(self.predict(x), y)
			self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

		self.q /= self.q.sum()

		loss = torch.dot(losses, self.q) / len(minibatches)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return {'loss': loss.item()}


class MLDG(ERM):
	"""
	Model-Agnostic Meta-Learning
	Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
	Related: https://arxiv.org/pdf/1703.03400.pdf
	Related: https://arxiv.org/pdf/1910.13580.pdf
	TODO: update() has at least one bug, possibly more. Disabling this whole
	algorithm until it gets figured out.
	"""
	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(MLDG, self).__init__(input_shape, num_classes, num_domains,
								   hparams)

	def update(self, minibatches):
		"""
		Terms being computed:
			* Li = Loss(xi, yi, params)
			* Gi = Grad(Li, params)
			* Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
			* Gj = Grad(Lj, params)
			* params = Optimizer(params, Grad(Li + beta * Lj, params))
			*		 = Optimizer(params, Gi + beta * Gj)
		That is, when calling .step(), we want grads to be Gi + beta * Gj
		For computational efficiency, we do not compute second derivatives.
		"""
		num_mb = len(minibatches)
		objective = 0

		self.optimizer.zero_grad()
		for p in self.network.parameters():
			if p.grad is None:
				p.grad = torch.zeros_like(p)

		for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
			# fine tune clone-network on task "i"
			inner_net = copy.deepcopy(self.network)

			inner_opt = torch.optim.Adam(
				inner_net.parameters(),
				lr=self.hparams["lr"],
				weight_decay=self.hparams['weight_decay']
			)

			inner_obj = F.cross_entropy(inner_net(xi), yi)

			inner_opt.zero_grad()
			inner_obj.backward()
			inner_opt.step()

			# The network has now accumulated gradients Gi
			# The clone-network has now parameters P - lr * Gi
			for p_tgt, p_src in zip(self.network.parameters(),
									inner_net.parameters()):
				if p_src.grad is not None:
					p_tgt.grad.data.add_(p_src.grad.data / num_mb)

			# `objective` is populated for reporting purposes
			objective += inner_obj.item()

			# this computes Gj on the clone-network
			loss_inner_j = F.cross_entropy(inner_net(xj), yj)
			grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
				allow_unused=True)

			# `objective` is populated for reporting purposes
			objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

			for p, g_j in zip(self.network.parameters(), grad_inner_j):
				if g_j is not None:
					p.grad.data.add_(
						self.hparams['mldg_beta'] * g_j.data / num_mb)

			# The network has now accumulated gradients Gi + beta * Gj
			# Repeat for all train-test splits, do .step()

		objective /= len(minibatches)

		self.optimizer.step()

		return {'loss': objective}

	# This commented "update" method back-propagates through the gradients of
	# the inner update, as suggested in the original MAML paper.  However, this
	# is twice as expensive as the uncommented "update" method, which does not
	# compute second-order derivatives, implementing the First-Order MAML
	# method (FOMAML) described in the original MAML paper.

	# def update(self, minibatches):
	#	  objective = 0
	#	  beta = self.hparams["beta"]
	#	  inner_iterations = self.hparams["inner_iterations"]

	#	  self.optimizer.zero_grad()

	#	  with higher.innerloop_ctx(self.network, self.optimizer,
	#		  copy_initial_weights=False) as (inner_network, inner_optimizer):

	#		  for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
	#			  for inner_iteration in range(inner_iterations):
	#				  li = F.cross_entropy(inner_network(xi), yi)
	#				  inner_optimizer.step(li)
	#
	#			  objective += F.cross_entropy(self.network(xi), yi)
	#			  objective += beta * F.cross_entropy(inner_network(xj), yj)

	#		  objective /= len(minibatches)
	#		  objective.backward()
	#
	#	  self.optimizer.step()
	#
	#	  return objective


class AbstractMMD(ERM):
	"""
	Perform ERM while matching the pair-wise domain feature distributions
	using MMD (abstract class)
	"""
	def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
		super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
								  hparams)
		if gaussian:
			self.kernel_type = "gaussian"
		else:
			self.kernel_type = "mean_cov"

	def my_cdist(self, x1, x2):
		x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
		x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
		res = torch.addmm(x2_norm.transpose(-2, -1),
						  x1,
						  x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
		return res.clamp_min_(1e-30)

	def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
										   1000]):
		D = self.my_cdist(x, y)
		K = torch.zeros_like(D)

		for g in gamma:
			K.add_(torch.exp(D.mul(-g)))

		return K

	def mmd(self, x, y):
		if self.kernel_type == "gaussian":
			Kxx = self.gaussian_kernel(x, x).mean()
			Kyy = self.gaussian_kernel(y, y).mean()
			Kxy = self.gaussian_kernel(x, y).mean()
			return Kxx + Kyy - 2 * Kxy
		else:
			mean_x = x.mean(0, keepdim=True)
			mean_y = y.mean(0, keepdim=True)
			cent_x = x - mean_x
			cent_y = y - mean_y
			cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
			cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

			mean_diff = (mean_x - mean_y).pow(2).mean()
			cova_diff = (cova_x - cova_y).pow(2).mean()

			return mean_diff + cova_diff

	def update(self, minibatches):
		objective = 0
		penalty = 0
		nmb = len(minibatches)

		features = [self.featurizer(xi) for xi, _ in minibatches]
		classifs = [self.classifier(fi) for fi in features]
		targets = [yi for _, yi in minibatches]

		for i in range(nmb):
			objective += F.cross_entropy(classifs[i], targets[i])
			for j in range(i + 1, nmb):
				penalty += self.mmd(features[i], features[j])

		objective /= nmb
		if nmb > 1:
			penalty /= (nmb * (nmb - 1) / 2)

		self.optimizer.zero_grad()
		(objective + (self.hparams['mmd_gamma']*penalty)).backward()
		self.optimizer.step()

		if torch.is_tensor(penalty):
			penalty = penalty.item()

		return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
	"""
	MMD using Gaussian kernel
	"""

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(MMD, self).__init__(input_shape, num_classes,
										  num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
	"""
	MMD using mean and covariance difference
	"""

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(CORAL, self).__init__(input_shape, num_classes,
										 num_domains, hparams, gaussian=False)
