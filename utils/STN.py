import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer_2d(nn.Module):
    def __init__(self,islabel):
        super(SpatialTransformer_2d, self).__init__()
        self.islabel = islabel
    def forward(self, src, flow,mode = 'bilinear'):
        if(self.islabel):
            mode = 'nearest'

        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape] 
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  
        grid = torch.unsqueeze(grid, 0)  
        grid = grid.type(torch.FloatTensor)
        grid = grid.cuda()

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        output = nnf.grid_sample(src, new_locs, mode=mode)
        if(self.islabel):
            output[::,0]=1-output[::,1]
        return output

    
    
class Re_SpatialTransformer_2d(nn.Module):
    def __init__(self,islabel):
        super(Re_SpatialTransformer_2d, self).__init__()
        self.islabel =islabel
        self.stn = SpatialTransformer_2d(self.islabel)

    def forward(self, src, flow, mode='bilinear'):
        flow = -1 * self.stn(flow, flow, mode='bilinear')

        return self.stn(src, flow, mode)