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
        # https://learn.adafruit.com/led-tricks-gamma-correction/the-quick-fix
        self.gamma8 = torch.Tensor(
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,
         2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,
         5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10,
         10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16,
         17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25,
         25, 26, 27, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36,
         37, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 50,
         51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68,
         69, 70, 72, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89,
         90, 92, 93, 95, 96, 98, 99,101,102,104,105,107,109,110,112,114,
        115,117,119,120,122,124,126,127,129,131,133,135,137,138,140,142,
        144,146,148,150,152,154,156,158,160,162,164,167,169,171,173,175,
        177,180,182,184,186,189,191,193,196,198,200,203,205,208,210,213,
        215,218,220,223,225,228,231,233,236,239,241,244,247,249,252,255])
        # https://stackoverflow.com/questions/69424870/how-to-write-a-kernel-for-pytorch-convolution-function
        self.kernel = torch.Tensor([[-0.07,-0.06,-0.05,-0.06,-0.07],
                                    [-0.06,0.04,0.14,0.04,-0.06],
                                    [-0.05,0.14,0.28,0.14,-0.05],
                                    [-0.06,0.04,0.14,0.04,-0.06],
                                    [-0.07,-0.06,-0.05,-0.06,-0.07]])
        self.kernel = self.kernel.unsqueeze(0)
        self.kernel = torch.cat([self.kernel,-self.kernel],0)
        self.kernel = self.kernel.unsqueeze(1)

    def __call__(self, tensor):
        # Use finer granularity if input max isn't too large
        maxt                                    = torch.max(tensor)

        spike_tensor = tensor.clone()
        # translate raw data into indices of the nonlinear intensity of gamma8
        spike_tensor = (spike_tensor / maxt * (self.gamma8.size()[0] - 1)).long()
        conv_in = spike_tensor.unsqueeze(1)
        #print(conv_in.size())
        #print(self.kernel.size())
        spike_tensor = F.conv2d(conv_in.float(), self.kernel, padding="same", stride=1)
        # from indices to gamma8 intensity
        # spike_tensor = self.gamma8[spike_tensor]
        # delay: 0-gamma -> value: 0-maxt
        # granularity is maxt/gamma
        # spike_tensor = (self.gamma8.size()[0]) - self.gamma8[spike_tensor]
        # spike_tensor = spike_tensor / len(self.gamma8)
        spike_tensor = self.gamma_time - spike_tensor / (self.gamma8.size()[0]) * self.gamma_time
        # spike_tensor[spike_tensor > (self.gamma_time - 1)] = float('Inf')
        # print(spike_tensor.squeeze().size())
        # input()
        return spike_tensor.squeeze()


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
