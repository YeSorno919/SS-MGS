import torch.nn as nn
import torch.nn.functional as F
import torch
__all__ = ['GumbelTopK', 'TensorBuffer','RNSF_ContrastiveLoss','ContrastiveLoss_Random_Sampling','ContrastiveLoss_Mean_Sampling']
class GumbelTopK(nn.Module):

    def __init__(self, k: int, dim: int = -1, gumble: bool = False):
        super().__init__()
        self.k = k
        self.dim = dim
        self.gumble = gumble

    def forward(self, logits):

        if self.gumble:
            u = torch.rand(size=logits.shape, device=logits.device)
            z = - torch.log(- torch.log(u))
            return torch.topk(logits + z, self.k, dim=self.dim)
        else:
            a = torch.topk(logits, self.k, dim=self.dim)
            return torch.topk(logits, self.k, dim=self.dim)

class TensorBuffer:
    def __init__(self, buffer_size: int, concat_dim: int, retain_gradient: bool = True):
        self.buffer_size = buffer_size
        self.concat_dim = concat_dim
        self.retain_gradient = retain_gradient
        self.tensor_list = []

    def update(self, tensor):
        if len(self.tensor_list) >= self.buffer_size:
            self.tensor_list.pop(0)
        if self.retain_gradient:
            self.tensor_list.append(tensor)
        else:
            self.tensor_list.append(tensor.detach())

    @property
    def values(self):
        return torch.cat(self.tensor_list, dim=self.concat_dim)

class RNSF_ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, sample_num: int = 50):
        super().__init__()
        self.tau = temperature
        self.sample_num = sample_num 
        self.topk = GumbelTopK(k=sample_num, dim=0) 
    def check_input(self, input_logits, target_logits, input_seg, target_seg):

        if input_logits is not None and target_logits is not None:
            assert input_seg is None and target_seg is None, \
                'When the logits are provided, do not provide segmentation.'
            if input_logits.shape[1] == 1: 
                input_max_probs = torch.sigmoid(input_logits)
                target_max_probs = torch.sigmoid(target_logits)

                input_cls_seg = (input_max_probs > 0.5).to(torch.float32)
                target_cls_seg = (target_max_probs > 0.5).to(torch.float32)
            else:  
                input_max_probs, input_cls_seg = torch.max(F.softmax(input_logits, dim=1), dim=1)
                target_max_probs, target_cls_seg = torch.max(F.softmax(target_logits, dim=1), dim=1)
            return input_cls_seg, target_cls_seg, input_max_probs, target_max_probs
        elif input_seg is not None and target_seg is not None:
            assert input_logits is None and target_logits is None, \
                'When the segmentation are provided, do not provide logits.'
            confident_probs = torch.ones_like(input_seg)
            return input_seg, target_seg, confident_probs, confident_probs
        else:
            raise ValueError('The logits/segmentation are not paired, please check the input')

    def forward(self, input, positive, negative, input_logits=None, negative_logits=None, input_seg=None, negative_seg=None):
        B, C, *spatial_size = input.shape  # N = H * W * D
        spatial_dims = len(spatial_size)

        # input/target shape: B * N, C 
        norm_input = F.normalize(input.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1) 
        norm_positive = F.normalize(positive.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        norm_negative = F.normalize(negative.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)

        seg_input, seg_negative, input_prob, negative_prob = self.check_input(
            input_logits, negative_logits, input_seg, negative_seg)


        seg_input = seg_input.flatten()  # B * N
        seg_negative = seg_negative.flatten()  # B * N
        negative_prob = negative_prob.flatten()  # B * N


        positive = F.cosine_similarity(norm_input, norm_positive, dim=-1)  # 


        diff_cls_matrix = (seg_input.unsqueeze(0) != seg_negative.unsqueeze(1)).to(torch.float16)  # B * N, B * N #

        prob_matrix = negative_prob.unsqueeze(1).expand_as(diff_cls_matrix)  # B * N, B * N  

        masked_target_prob_matrix = diff_cls_matrix * prob_matrix  # mask positive pairs, sample negative only

 
        sampled_negative_indices = self.topk(masked_target_prob_matrix).indices  # K, B * N = 
 
        sampled_negative = norm_negative[sampled_negative_indices]  # K, B * N, C 

        negative_sim_matrix = F.cosine_similarity(norm_input.unsqueeze(0).expand_as(sampled_negative), 
                                                  sampled_negative, dim=-1)  # K, B * N

        nominator = torch.exp(positive / self.tau) 
        denominator = torch.exp(negative_sim_matrix / self.tau).sum(dim=0) + nominator 
        loss = -torch.log(nominator / (denominator + 1e-8)).mean()

        alter_negative_sim_matrix = F.cosine_similarity(norm_positive.unsqueeze(0).expand_as(sampled_negative),
                                                            sampled_negative, dim=-1)  # K, B * N
        alter_denominator = torch.exp(alter_negative_sim_matrix / self.tau).sum(dim=0) + nominator
        alter_loss = -torch.log(nominator / (alter_denominator + 1e-8)).mean()
        
        loss = loss + alter_loss
        return loss

# Randmom Sampling Ablation
class ContrastiveLoss_Random_Sampling(nn.Module):
    def __init__(self, temperature: float = 0.07, sample_num: int = 50):
        super().__init__()
        self.tau = temperature
        self.sample_num = sample_num #

    def check_input(self, input_logits, target_logits, input_seg, target_seg):

        if input_logits is not None and target_logits is not None:
            assert input_seg is None and target_seg is None, \
                'When the logits are provided, do not provide segmentation.'
            if input_logits.shape[1] == 1: 
                input_max_probs = torch.sigmoid(input_logits)
                target_max_probs = torch.sigmoid(target_logits)

                input_cls_seg = (input_max_probs > 0.5).to(torch.float32)
                target_cls_seg = (target_max_probs > 0.5).to(torch.float32)
            else:  
                input_max_probs, input_cls_seg = torch.max(F.softmax(input_logits, dim=1), dim=1)
                target_max_probs, target_cls_seg = torch.max(F.softmax(target_logits, dim=1), dim=1)
            return input_cls_seg, target_cls_seg, input_max_probs, target_max_probs
        elif input_seg is not None and target_seg is not None:
            assert input_logits is None and target_logits is None, \
                'When the segmentation are provided, do not provide logits.'
            confident_probs = torch.ones_like(input_seg)
            return input_seg, target_seg, confident_probs, confident_probs
        else:
            raise ValueError('The logits/segmentation are not paired, please check the input')

    def forward(self, input, positive, negative, input_logits=None, negative_logits=None, input_seg=None, negative_seg=None):

        B, C, *spatial_size = input.shape  # N = H * W * D 
        spatial_dims = len(spatial_size)

        # input/target shape: B * N, C 
        norm_input = F.normalize(input.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1) 
        norm_positive = F.normalize(positive.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        norm_negative = F.normalize(negative.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)

        seg_input, seg_negative, input_prob, negative_prob = self.check_input(
            input_logits, negative_logits, input_seg, negative_seg)

        seg_input = seg_input.flatten()  # B * N
        seg_negative = seg_negative.flatten()  # B * N

        negative_prob = negative_prob.flatten()  # B * N


        positive = F.cosine_similarity(norm_input, norm_positive, dim=-1)  # B * N

        diff_cls_matrix = (seg_input.unsqueeze(0) != seg_negative.unsqueeze(1)).to(torch.float16)  # B * N, B * N 

        nonzero_indices = torch.nonzero(diff_cls_matrix[:, 0], as_tuple=True)[0]

        sampled_indices = nonzero_indices[torch.randperm(len(nonzero_indices), device=diff_cls_matrix.device)[:self.sample_num if self.sample_num<len(nonzero_indices)  else len(nonzero_indices) ]]
        sampled_negative_indices = sampled_indices.unsqueeze(1).expand(self.sample_num if self.sample_num<len(nonzero_indices)  else len(nonzero_indices) , seg_input.size()[0])


        sampled_negative = norm_negative[sampled_negative_indices]  # K, B * N, C 

        negative_sim_matrix = F.cosine_similarity(norm_input.unsqueeze(0).expand_as(sampled_negative), 
                                                  sampled_negative, dim=-1)  # K, B * N
        nominator = torch.exp(positive / self.tau) # 
        denominator = torch.exp(negative_sim_matrix / self.tau).sum(dim=0) + nominator #
        loss = -torch.log(nominator / (denominator + 1e-8)).mean()
        
        alter_negative_sim_matrix = F.cosine_similarity(norm_positive.unsqueeze(0).expand_as(sampled_negative),
                                                            sampled_negative, dim=-1)  # K, B * N
        alter_denominator = torch.exp(alter_negative_sim_matrix / self.tau).sum(dim=0) + nominator
        alter_loss = -torch.log(nominator / (alter_denominator + 1e-8)).mean()
        loss = loss + alter_loss
        return loss
    
# Mean Sampling Ablation
class ContrastiveLoss_Mean_Sampling(nn.Module):
    def __init__(self, temperature: float = 0.07, sample_num: int = 50):
        super().__init__()
        self.tau = temperature
        self.sample_num = sample_num #

    def check_input(self, input_logits, target_logits, input_seg, target_seg):

        if input_logits is not None and target_logits is not None:
            assert input_seg is None and target_seg is None, \
                'When the logits are provided, do not provide segmentation.'
            if input_logits.shape[1] == 1: 
                input_max_probs = torch.sigmoid(input_logits)
                target_max_probs = torch.sigmoid(target_logits)

                input_cls_seg = (input_max_probs > 0.5).to(torch.float32)
                target_cls_seg = (target_max_probs > 0.5).to(torch.float32)
            else:  
                input_max_probs, input_cls_seg = torch.max(F.softmax(input_logits, dim=1), dim=1)
                target_max_probs, target_cls_seg = torch.max(F.softmax(target_logits, dim=1), dim=1)
            return input_cls_seg, target_cls_seg, input_max_probs, target_max_probs
        elif input_seg is not None and target_seg is not None:
            assert input_logits is None and target_logits is None, \
                'When the segmentation are provided, do not provide logits.'
            confident_probs = torch.ones_like(input_seg)
            return input_seg, target_seg, confident_probs, confident_probs
        else:
            raise ValueError('The logits/segmentation are not paired, please check the input')

    def forward(self, input, positive, negative, input_logits=None, negative_logits=None, input_seg=None, negative_seg=None):

        B, C, *spatial_size = input.shape  # N = H * W * D 
        spatial_dims = len(spatial_size)

        # input/target shape: B * N, C
        norm_input = F.normalize(input.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1) 
        norm_positive = F.normalize(positive.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        norm_negative = F.normalize(negative.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        seg_input, seg_negative, input_prob, negative_prob = self.check_input(
            input_logits, negative_logits, input_seg, negative_seg)

        seg_input = seg_input.flatten()  # B * N
        seg_negative = seg_negative.flatten()  # B * N
        negative_prob = negative_prob.flatten()  # B * N

        positive = F.cosine_similarity(norm_input, norm_positive, dim=-1)  # B * N

        diff_cls_matrix = (seg_input.unsqueeze(0) != seg_negative.unsqueeze(1)).to(torch.float16)  # B * N, B * N 
        # 构建负样本概率矩阵
        prob_matrix = negative_prob.unsqueeze(1).expand_as(diff_cls_matrix)  # B * N, B * N  

        nonzero_indices = torch.nonzero(diff_cls_matrix[:, 0], as_tuple=True)[0]

        sampled_indices = nonzero_indices[torch.randperm(len(nonzero_indices), device=diff_cls_matrix.device)[:self.sample_num if self.sample_num<len(nonzero_indices)  else len(nonzero_indices) ]]
        sampled_negative_indices = sampled_indices.unsqueeze(1).expand(self.sample_num if self.sample_num<len(nonzero_indices)  else len(nonzero_indices) , seg_input.size()[0])

        # Mean sampling
        sampled_negative = norm_negative[sampled_negative_indices].mean(dim=0).unsqueeze(0)  # K, B * N, C 

        negative_sim_matrix = F.cosine_similarity(norm_input.unsqueeze(0).expand_as(sampled_negative), 
                                                  sampled_negative, dim=-1)  # K, B * N

        nominator = torch.exp(positive / self.tau) 
        denominator = torch.exp(negative_sim_matrix / self.tau).sum(dim=0) + nominator 
        loss = -torch.log(nominator / (denominator + 1e-8)).mean()

        alter_negative_sim_matrix = F.cosine_similarity(norm_positive.unsqueeze(0).expand_as(sampled_negative),
                                                            sampled_negative, dim=-1)  # K, B * N
        alter_denominator = torch.exp(alter_negative_sim_matrix / self.tau).sum(dim=0) + nominator
        alter_loss = -torch.log(nominator / (alter_denominator + 1e-8)).mean()
        loss = loss + alter_loss
        return loss    
