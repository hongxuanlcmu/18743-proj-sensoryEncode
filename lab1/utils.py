### Importing libraries ###

import torch
import torch.nn as nn
import torch.nn.functional as F


### PosNeg Encoding ###

class PosNeg(object):
    def __init__(self, pn_threshold):
        self.pn_thresh = pn_threshold

    def __call__(self, tensor):
        maxt                                    = torch.max(tensor)
        tensor[tensor >= (self.pn_thresh*maxt)] = float('Inf')
        tensor[tensor < (self.pn_thresh*maxt)]  = 0
        tensor_pos                              = tensor.clone()
        tensor_pos[tensor_pos == 0]             = 1
        tensor_pos[tensor_pos == float('Inf')]  = 0
        tensor_pos[tensor_pos == 1]             = float('Inf')
        out                    = torch.cat([tensor_pos,tensor], dim=0)
        return out
    
## Intensity to latency encoding
class IntensityTranslation(object):
    def __init__(self, gamma_time):
        self.gamma_time = gamma_time
        pass

    def __call__(self, tensor):
        # Use finer granularity if input max isn't too large
        maxt                                    = torch.max(tensor)
        # renormalize tensor to use the finer granularity
        tensor = tensor / maxt
        # copy this to use as output
        # original tensor used for compare
        spike_tensor = tensor.clone()
        # delay: 0-gamma -> value: 0-maxt
        # granularity is maxt/gamma
        spike_tensor = self.gamma_time - spike_tensor * self.gamma_time
        # spike_tensor[spike_tensor > (self.gamma_time - 1)] = float('Inf')
        return spike_tensor


class DualTNNVoterTallyLayer(nn.Module):
    def __init__(self, rows, cols, nprev, num_classes, thetav_lo, thetav_hi, tau_eff=3, wres=3, w_init="half",\
                 device="cpu"):
        super(DualTNNVoterTallyLayer, self).__init__()
        self.rows          = rows
        self.cols          = cols
        self.p             = nprev
        self.q             = num_classes
        self.tau_eff       = tau_eff

        self.wmax          = 2**wres - 1

        if w_init       == "zero":
            self.weights   = nn.Parameter(torch.zeros(2,self.rows*self.cols*self.p,self.q,self.tau_eff+1),\
                                          requires_grad=False)
        elif w_init     == "half":
            self.weights   = nn.Parameter((self.wmax/2)*torch.ones(2,self.rows*self.cols*self.p,self.q,\
                                                                   self.tau_eff+1),requires_grad=False)
        elif w_init     == "uniform":
            self.weights   = nn.Parameter(torch.randint(low=0, high=self.wmax+1, \
                                                        size=(2,self.rows*self.cols*self.p,self.q,self.tau_eff+1))\
                                          .type(torch.FloatTensor), requires_grad=False)
        elif w_init     == "normal":
            self.weights   = nn.Parameter(torch.round(((self.wmax+1)/2+torch.randn(2,self.rows*self.cols*self.p,\
                                                                                   self.q,self.tau_eff+1))\
                                                      .clamp_(0,self.wmax)), requires_grad=False)

        self.stdp          = DualRSTDP_Voter(wres, thetav_lo, thetav_hi,\
                                             self.rows, self.cols, self.p, self.q, self.tau_eff, device=device)

        self.const         = torch.arange(self.tau_eff+1).repeat(2,self.rows*self.cols*self.p,self.q,1).to(device)
        self.zeros         = torch.zeros(2,self.rows*self.cols*self.p,self.q,self.tau_eff+1).to(device)
        self.szr           = torch.zeros(self.q).to(device)
        self.son           = torch.ones(self.q).to(device)

    def __call__(self, input_spikes):
        self.votes                     = self.zeros.clone()

        voter_in                       = input_spikes.unsqueeze(3).repeat(1,1,1,self.q).reshape(-1,self.q)\
                                                            .unsqueeze(0).repeat(2,1,1)

        cond1                          = (voter_in >= self.tau_eff)
        cond2                          = (voter_in != float('Inf'))
        voter_in[cond1 * cond2]        = self.tau_eff

        voter_in                       = voter_in.unsqueeze(3).repeat(1,1,1,self.tau_eff+1)
        voter_in                       = self.const - voter_in
        voter_in[voter_in!=0]          = -1
        voter_in                       += 1

        sel_weights                    = voter_in * self.weights
        cond3                          = (sel_weights >= (self.wmax/2))
        self.votes[cond3]              = 1
        votes                          = self.votes.permute(2,0,1,3).reshape(self.q,-1)

        self.prediction                = self.szr.clone()
        self.tally                     = torch.sum(votes,dim=1)
        self.prediction.scatter_(0,torch.argmax(self.tally,dim=0,keepdim=True),self.son)

        return self.prediction, voter_in[0,:,:,:], self.votes


class DualRSTDP_Voter():
    def __init__(self, wres, thetav_lo, thetav_hi, rows, cols, p, q, tau_eff, device="cpu"):
        self.wmax           = 2**(wres)-1
        self.thetav_lo      = thetav_lo
        self.thetav_hi      = thetav_hi
        self.num            = rows * cols * p
        self.q              = q
        self.tau_eff        = tau_eff
        self.constant       = torch.arange(self.q).to(device)

    def __call__(self, target, voter_in, weights):
        target            = self.constant - target.squeeze().repeat(self.q)
        target[target!=0] = -1
        target            += 1
        target            = target.unsqueeze(0).repeat(self.num,1).unsqueeze(2).repeat(1,1,self.tau_eff+1)

        weights[0,:,:,:] -= voter_in * self.thetav_lo
        weights[0,:,:,:] += voter_in * target
        weights[1,:,:,:] -= voter_in * self.thetav_hi
        weights[1,:,:,:] += voter_in * target

        return nn.Parameter(weights.clamp_(0, self.wmax), requires_grad=False)
