# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import numpy as np
import torch
import torch.utils.data
from itertools import chain

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader
import torch.nn.functional as F
    
def svd(self, x, y):
	sorted_data = [[] for _ in range(self.num_classes)]
	for i, lbl in enumerate(y):
		sorted_data[lbl].append(x[i])
	sorted_data = [torch.stack(class_data).cpu() for class_data in sorted_data]

	components = [[] for i in range(self.num_classes)]
	singular_values = [[] for i in range(self.num_classes)]

	for j in range(self.num_classes):
		u,s,vt = torch.svd(sorted_data[j])
		components[j] = vt.t()[:self.hparams['n_comp']]
		singular_values[j] = s[:self.hparams['n_comp']]

	return components, singular_values

def sorted_features(featurizer, num_classes,class_loader):
	"""Sort dataset based on classes.

	Parameters:
		data (np.ndarray): data array
		labels (np.ndarray): one dimensional array of class labels
		num_classes (int): number of classes
		stack (bol): combine sorted data into one numpy array

	Return:
		sorted data (np.ndarray), sorted_labels (np.ndarray)

	"""
	c = [[] for _ in range(num_classes)]
	for x,y in class_loader:
		batch = featurizer(x.cuda()).cpu().detach()
		for i in range(len(batch)):
			c[y[i]].append(batch[i])
	c = [torch.stack(class_data,0) for class_data in c]
	c = [F.normalize(class_data - torch.mean(class_data, 0)) for class_data in c]
	return c

def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.cuda()
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

def mutual_information_per_class(featurizer,num_classes, class1, class2):
	c1 = class1
	c2 = class2
	mi = []
	for i in range(num_classes):
		z = torch.cat((c1[i],c2[i]), 0)
		m,p = z.shape
		m1, _ = c1[i].shape
		m2, _ = c2[i].shape
		I = torch.eye(p).cpu()
		scalar = p / (m * 0.5)
		scalar1 = p / (m1 * 0.5)
		scalar2 = p / (m2 * 0.5)
		ld = torch.logdet(I + scalar * (z.T).matmul(z)) / 2.
		ld1 = m1 * torch.logdet(I + scalar1 * (c1[i].T).matmul(c1[i])) / (2. * m)
		ld2 = m2 * torch.logdet(I + scalar2 * (c2[i].T).matmul(c2[i])) / (2. * m)
		mi.append((ld - ld1 - ld2).item())
	return mi


args = {}
args['dataset'] = 'RotatedMNIST'
args['test_envs'] = [0]
args['output_dir'] = 'output/mnist_tests/mcr_env0_fd40'
hparams = hparams_registry.default_hparams('MCR', args['dataset'])
fd = 40
hparams['beta'] = 0

start_step = 0
algorithm_dict = None

print('HParams:')
for k, v in sorted(hparams.items()):
    print('\t{}: {}'.format(k, v))

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if args['dataset'] in vars(datasets):
    dataset = vars(datasets)[args['dataset']]('data',
        args['test_envs'], hparams)
else:
    raise NotImplementedError
if args['dataset'] == 'RotatedMNIST':
    num_classes = 10.0
if args['dataset'] == 'PACS':
    num_classes = 7.0

# Split each env into an 'in-split' and an 'out-split'. We'll train on
# each in-split except the test envs, and evaluate on all splits.
in_splits = []
out_splits = []
for env_i, env in enumerate(dataset):
    out, in_ = misc.split_dataset(env,
        int(len(env)*0.2),
        misc.seed_hash(0, env_i))
    if hparams['class_balanced']:
        in_weights = misc.make_weights_for_balanced_classes(in_)
        out_weights = misc.make_weights_for_balanced_classes(out)
    else:
        in_weights, out_weights = None, None
    in_splits.append((in_, in_weights))
    out_splits.append((out, out_weights))

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

algorithm_class = algorithms.get_algorithm_class('MCR')
hparams['fd'] = fd
hparams['n_comp'] = int(hparams['fd']/num_classes)
algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
    len(dataset) - len(args['test_envs']), hparams)

algorithm.load_state_dict(torch.load(args['output_dir'] + '/G.pth.tar'))
algorithm = algorithm.to(device)
algorithm.eval()
all_data = chain(*[eval_loaders[i] for i in range(len(in_splits)) if i not in args['test_envs']])
algorithm.update(all_data, components=True)
    
print('SVD Accuracy')
evals = zip(eval_loader_names, eval_loaders, eval_weights)
results = {}
for name, loader, weights in evals:
    acc = misc.accuracy(algs, loader, weights, device)
    results[name+'_acc'] = acc
results_keys = sorted(results.keys())
misc.print_row(results_keys, colwidth=12)
misc.print_row([results[key] for key in results_keys], colwidth=12)

t = args['test_envs'][0]
print('SVD by Domain')
count = 0
components = {}
singular_values = {}
sorted_data = {}
for j in range(len(in_splits)):
    if j != t:
        p = []
        all_y = []
        iterator = iter(eval_loaders[j])
        for x,y in iterator:
            p.append(algorithm.featurizer(x.cuda()).cpu().detach())
            all_y.append(y)
        p, y = torch.cat(p), torch.cat(all_y)
        sorted_data[j] = [[] for _ in range(algorithm.num_classes)]
        for i, lbl in enumerate(y):
            sorted_data[j][lbl].append(p[i])
        sorted_data[j] = [torch.stack(class_data).cpu() for class_data in sorted_data[j]]
        
        components[j] = {}
        singular_values[j] = {}
        for c in range(algorithm.num_classes):
            u,s,vt = torch.svd(sorted_data[j][c])
            components[j][c] = vt.t()[:1]
            singular_values[j][c] = s[:1]
    else:
        accuracy = 0
        count = 0
        p = []
        all_y = []
        iterator = iter(eval_loaders[j])
        for x,y in iterator:
            p.append(algorithm.featurizer(x.cuda()).cpu().detach())
            prediction = algorithm.predict(x)
            all_y.append(prediction)
            accuracy += np.sum(prediction == y)
            count += len(y)
        print('Original Accuracy: %f'%(float(accuracy)/count))
        p, y = torch.cat(p), torch.cat(all_y)
        sorted_data[j] = [[] for _ in range(algorithm.num_classes)]
        for i, lbl in enumerate(y):
            sorted_data[j][lbl].append(p[i])
        sorted_data[j] = [torch.stack(class_data).cpu() for class_data in sorted_data[j]]
        
            
print('Mutual Information per Class')
mutual_info = {}
for j in range(len(in_splits)):
    mutual_info[j] = mutual_information_per_class(featurizer, algorithm.num_classes,sorted_data[j], sorted_data[t])

components = {}
singular_values = {}
sorted_data = {}
def predict(a, x):
    x = a.featurizer(x)
    scores_svd = []
    for j in range(a.num_classes):
        svd_j = torch.matmul(torch.matmul(F.normalize(a.singular_values[j], dim=0)*a.components[j].t(),a.components[j]).to(device),x.t().to(device))
        score_svd_j = torch.norm(svd_j, dim=0)
        
        finite_svd_j = 0
        for i in range(a.num_domains):
            if i != t:
                svd_j = torch.matmul(torch.matmul(F.normalize(singular_values[i][j], dim=0)*components[i][j].t(),components[i][j]).to(device),x.t().to(device))
                finite_svd_j += (1/mutual_info[i][j])* torch.norm(svd_j, dim=0)
                
        print(score_svd_j/finite_svd_j)
       
        scores_svd.append(score_svd_j + finite_svd_j)
        p = torch.argmax(torch.stack(scores_svd), dim=0)
    return F.one_hot(p, a.num_classes)
    
    
accuracy = 0
count = 0
iterator = iter(eval_loaders[t])
for x,y in iterator:
    prediction = predict(algorithm, x)
    accuracy += np.sum(prediction == y)
    count += len(y)
print('Improved Accuracy: %f'%(float(accuracy)/count))
