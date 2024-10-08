import torch
import os
import sklearn.neighbors as nn
import numpy as np


def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if inds.numel() == 1:
        if inds.item() == val:
            return True
    return False

def na():  # shorthand for new axis
    return torch.unsqueeze(torch.tensor(0), dim=0)

def flatten_nd_array(pts_nd, axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = torch.tensor(pts_nd.shape)
    nax = torch.tensor([i for i in range(NDIM) if i != axis])
    NPTS = torch.prod(SHP[nax]).item()
    axorder = torch.cat((nax, torch.tensor([axis])), dim=0)
    pts_flt = pts_nd.permute(axorder.tolist())
    pts_flt = pts_flt.reshape(NPTS, SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt, pts_nd, axis=1, squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = torch.tensor(pts_nd.shape)
    nax = torch.tensor([i for i in range(NDIM) if i != axis])
    NPTS = torch.prod(SHP[nax]).item()

    if squeeze:
        axorder = nax
        axorder_rev = torch.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.permute(axorder_rev.tolist())
    else:
        axorder = torch.cat((nax, torch.tensor([axis])), dim=0)
        axorder_rev = torch.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.permute(axorder_rev.tolist())

    return pts_out


class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self, NN, sigma, km_filepath='', cc=-1):

        if check_value(torch.tensor(cc), -1):

            self.cc = np.load(km_filepath)
            # self.cc = torch.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)
        self.alreadyUsed = False

    def encode_points_mtx_nd(self, pts_nd, axis=1, returnSparse=False, sameBlock=True):
        pts_flt = flatten_nd_array(pts_nd, axis=axis)

        P = pts_flt.shape[0]
        if sameBlock and self.alreadyUsed:
            self.pts_enc_flt[...] = 0  # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = torch.zeros((P, self.K), dtype=torch.float32)
            self.p_inds = torch.arange(0, P, dtype=torch.int64).unsqueeze(1).expand(P, self.NN)

        (dists, inds) = self.nbrs.kneighbors(pts_flt.numpy())

        wts = torch.exp(-torch.tensor(dists) ** 2 / (2 * self.sigma ** 2))
        wts = (wts / torch.sum(wts, dim=1).unsqueeze(1).expand(P, self.NN))
        wts = torch.tensor(wts, dtype=torch.float32)

        self.pts_enc_flt[self.p_inds, torch.tensor(inds, dtype=self.p_inds.dtype)] = wts
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)

        return pts_enc_nd

    def decode_points_mtx_nd(self, pts_enc_nd, axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd, axis=axis)
        pts_dec_flt = torch.matmul(pts_enc_flt, self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt, pts_enc_nd, axis=axis)
        return pts_dec_nd

    def decode_1hot_mtx_nd(self, pts_enc_nd, axis=1, returnEncode=False):
        pts_1hot_nd = nd_argmax_1hot(pts_enc_nd, axis=axis)
        pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd, axis=axis)
        if returnEncode:
            return (pts_dec_nd, pts_1hot_nd)
        else:
            return pts_dec_nd


def _nnencode(data_ab_ss):
    '''Encode to 313bin
    Args:
        data_ab_ss: [N, H, W, 2]
    Returns:
        gt_ab_313 : [N, H, W, 313]
    '''
    NN = 10
    sigma = 5.0
    enc_dir = './resources/'

    data_ab_ss = torch.transpose(data_ab_ss, 1, 3)
    nnenc = NNEncode(NN, sigma, km_filepath=os.path.join(enc_dir, 'pts_in_hull.npy'))
    gt_ab_313 = nnenc.encode_points_mtx_nd(data_ab_ss, axis=1)

    gt_ab_313 = torch.transpose(gt_ab_313, 1, 3)
    return gt_ab_313


class PriorFactor():
    def __init__(self, alpha, gamma=0, verbose=False, priorFile=''):
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = torch.tensor(np.load(priorFile), )

        # define uniform probability
        self.uni_probs = torch.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs != 0] = 1.0
        self.uni_probs = self.uni_probs / torch.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution       
        self.prior_mix = (1 - self.gamma) * self.prior_probs + self.gamma * self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix ** -self.alpha
        self.prior_factor = self.prior_factor / torch.sum(self.prior_probs * self.prior_factor)  # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs * self.prior_factor
        self.implied_prior = self.implied_prior / torch.sum(self.implied_prior)  # re-normalize

        if self.verbose:
            self.print_correction_stats()

    def print_correction_stats(self):
        print('Prior factor correction:')
        print('  (alpha,gamma) = (%.2f, %.2f)' % (self.alpha, self.gamma))
        print('  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)' % (
            torch.min(self.prior_factor), torch.max(self.prior_factor), torch.mean(self.prior_factor),
            torch.median(self.prior_factor), torch.sum(self.prior_factor * self.prior_probs)))

    def forward(self, data_ab_quant, axis=1):
        data_ab_maxind = torch.argmax(data_ab_quant, dim=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        return corr_factor.unsqueeze(axis)


def _prior_boost(gt_ab_313):
    enc_dir = './resources'
    gamma = 0.5
    alpha = 1.0

    pc = PriorFactor(alpha, gamma, priorFile=os.path.join(enc_dir, 'prior_probs.npy'))

    gt_ab_313 = torch.transpose(gt_ab_313, 1, 3)
    prior_boost = pc.forward(gt_ab_313, axis=1)
    breakpoint()
    prior_boost = torch.transpose(prior_boost, 1, 3)
    return prior_boost

import warnings
import torch
import numpy as np
from PIL import Image
from skimage import color  # Ensure you have this for RGB to LAB conversion

def preprocess(image_paths):
    '''Preprocess
    Args: 
      image_paths: List of paths to images (N)
    Return:
      data_l: L channel batch (N * H * W * 1)
      gt_ab_313: ab discrete channel batch (N * H/4 * W/4 * 313)
      prior_boost_nongray: (N * H/4 * W/4 * 1) 
    '''
    warnings.filterwarnings("ignore")

    # Load images and convert to tensor
    images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
    data = torch.stack([torch.from_numpy(np.array(img)) for img in images])  # Shape: N x H x W x 3
    data = data.float() / 255.0  # Normalize to [0, 1]

    N, H, W, _ = data.shape

    # rgb2lab
    img_lab = color.rgb2lab(data.numpy())  # Convert to NumPy for rgb2lab

    # slice
    # L: [0, 100]
    img_l = torch.from_numpy(img_lab[:, :, :, 0:1])  # Convert back to tensor
    # ab: [-110, 110]
    data_ab = torch.from_numpy(img_lab[:, :, :, 1:])

    # scale img_l to [-50, 50]
    data_l = img_l - 50

    # subsample 1/4  (N * H/4 * W/4 * 2)
    data_ab_ss = data_ab[:, ::4, ::4, :]  # Subsampling

    # NonGrayMask {N, 1, 1, 1}
    thresh = 5
    nongray_mask = (torch.sum(torch.sum(torch.sum(torch.abs(data_ab_ss) > thresh, dim=1), dim=1), dim=1) > 0).unsqueeze(1).unsqueeze(2).unsqueeze(3)

    # NNEncoder
    # gt_ab_313: [N, H/4, W/4, 313]
    gt_ab_313 = _nnencode(data_ab_ss)

    # Prior_Boost 
    # prior_boost: [N, 1, H/4, W/4]
    prior_boost = _prior_boost(gt_ab_313)

    # Eltwise
    # prior_boost_nongray: [N, 1, H/4, W/4]
    prior_boost_nongray = prior_boost * nongray_mask

    return data_l, gt_ab_313, prior_boost_nongray



if __name__ == "__main__":
    #  loop through the dataset and preprocess
    image_paths = ['datasets/flower.jpg', 'datasets/flower2.jpg']
    data_l, gt_ab_313, prior_boost_nongray = preprocess(image_paths)
    print(data_l.shape, gt_ab_313.shape, prior_boost_nongray.shape)
