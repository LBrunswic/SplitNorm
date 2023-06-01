# SplitNorm


# TODO

## Pytorch Implementation
- [ ] Migrate code to Pytorch

## TF Implementation
- [x] Have single MNIST instance working
- [ ] Clean ConvKernel2 (Leo)
- [ ] Solve the fit training issue
##  Benchmarking
- [ ] Choice of datasets for showcase (2 scales of images datasets, Finance dataset, manifold learning)
- [ ] Choice of metrics for each (number of parameters, likelyhood, L2 norm, benefit)
- [ ] Review  of SOTA scores for each metrics/datasets

## Commanded layers
- [ ] reimplement based Dense/CNN layers with command kwarg

## Implementation of Flows
The goal is to have building blocks for flows
### GLOW/NICE
- [ ] Add

### Moser flow and other flow matching methods (https://proceedings.neurips.cc/paper/2021/hash/93a27b0bd99bac3e68a440b48aa421ab-Abstract.html)
- [ ] Implementation of Moser Flow
- [ ] Implement some regularization methods
- [ ] Bibliography on solution of poisson problem
###  FFJORD
- [ ] Add Benamou-Brenier regularization
- [ ] Better ODE solver

# Secondary normalization and generation
- [ ] implement a class designed for training of iterated normalization

# Theory
## Total Normalization loss
- [ ] Better computation of the Total normalization loss, in particular the channeled self-entropy
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
