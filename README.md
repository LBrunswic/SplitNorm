# SplitNorm

#HP TO TEST
## MNIST numbers
- [ ] variations on spatial encoding [0-16]
- [ ] variations on channel dimension [1-16]

# TODO
## Pytorch Implementation
- [ ] Migrate code to Pytorch

## TF Implementation
- [x] Have single MNIST instance working
- [x] Clean ConvKernel2 (Leo)
- [x] Solve the fit training issue
##  Benchmarking
- [ ] Choice of datasets for showcase (2 scales of images datasets, Finance dataset, manifold learning)
- [ ] Choice of metrics for each (number of parameters, likelyhood, L2 norm, benefit)
- [ ] Review  of SOTA scores for each metrics/datasets
### Benchmarking
- [ ] Write test script dedictated to MNIST/CIFAR

## Commanded layers
- [ ] reimplement based Dense/CNN layers with command kwarg


## Tests
### Basic tests
- [x] test script of convolutional kernel training (simple example)
- [x] test script of convolutional kernel training (involved example)
- [x] test script of HigherConvKernels with visualization of secondary normalization

### Reference architectures
- [x] Choice of a reference small architecture MNIST/CIFAR capable
- [x] Choice of a reference architecture for high res pictures  (Moser flow paper FFJORD ref)

### Description of test of influence of relative dimension of Channel space / distribution dimension
- [ ] Elementary script
- [ ] thorough tests
### Toward Publication tests
- [ ] Description of families of architecures to be tested and choices of hyperparameters
- [ ] Industrial test scripts

## Implementation of Flows

### Flow Constructors
Elementary NF are obtained by composing simple operations. Lifted normalization
via convolutional flows generalizes this to a channel space distributional parameterization.
The simplest instance is an Ensemble of NF (a finite family of $p$ flows, the channeller outputs a distribution hence a softmax weighting of the flows:
  $$ C(x) = \sum_{i=1}^p \alpha_{i} \delta_{c_i}, \quad \alpha ) \mathrm{softmax}(NN(x))$$


The constructions allow to
- [x] Ensemble of NF
- [x] Convolution of Flows (channeller,commander,kernel)

The goal is to have building blocks for flows
### GLOW/NICE
- [ ] Add

### Moser flow and other flow matching methods
[https://proceedings.neurips.cc/paper/2021/hash/93a27b0bd99bac3e68a440b48aa421ab-Abstract.html]
- [ ] Implementation of Moser Flow
- [ ] Implement some regularization methods
- [ ] Bibliography on solution of poisson problem
###  FFJORD
- [ ] Add Benamou-Brenier regularization


# Secondary normalization and generation
- [ ] implement a class designed for training of iterated normalization

# Theory
## Thorough comparison to other models
- [ ] Stability/expressivity analysis compared to NF including Matching flows (Moser, etc)
- [ ] How Higher Convolution compares to VAE/GAN especially regarding multi-modal distribution (why does those methods have mutli-mode issues why this is not the case here?)

## Total Normalization loss
- [x] Better computation of the Total normalization loss, in particular the channeled self-entropy
- [ ] Find regularization methods for continuously parameterized flows
- [ ] Dimensional analysis
## Intrinsic dataset dimension
- [ ] Bibliography review of dimensional methods and analysis
- [ ] add table to paper
- [ ] If need be, compute the intrinsic dimension of other datasets
## Iterated normalization
- [ ] Clarify situations in which it is useful to have more than one
## Wasserstein Gradient Flow training
- [ ] Formulate the correct training losses
## On AIGP
- [ ] Review possible usage of the method
- [ ] Context-Multi-task RL?
## On Optimal transport
- [ ] review of Wasserstein losses
- [ ] Brenier-McCann c-convex map flow parametrization
#
