import math

import torch
import torch.nn as nn
import torch.nn.functional as F



def accuracy(output, target, topk=(1,)):

	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	
	return res



# class DAMSoftmax(nn.Module):
#     def __init__(  # noqa: D107
#         self,
#         in_features: int,
#         out_features: int,
#         s: float = 64.0,
#         m: float = 0.5,
#         k: int = 128, 
#         eps: float = 1e-6,
#     ):
#         super(DAMSoftmax, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.k = k
#         self.eps = eps
        
#         self.weight = nn.Parameter(torch.FloatTensor(k, in_features, out_features))
#         nn.init.xavier_uniform_(self.weight)
        
#         self.fw = nn.Linear(1, 1)
#         nn.init.xavier_uniform_(self.fw.weight)
        
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.mse_fn  = nn.MSELoss()
#         self.softmax_fn = nn.Softmax()
#         self.smoothl1_fn = nn.SmoothL1Loss()
#         self.threshold  = math.pi - self.m
        
        

#     def __repr__(self) -> str:
#         """Object representation."""
#         rep = (
#             "DAMSoftmax("
#             f"in_features={self.in_features},"
#             f"out_features={self.out_features},"
#             f"s={self.s},"
#             f"m={self.m},"
#             f"k={self.k},"
#             f"eps={self.eps},"
#             f"loss_fn={self.loss_fn}"
#             ")"
#         )
#         return rep

    
#     def forward(self, input: torch.Tensor, factor: torch.Tensor, label: torch.LongTensor) -> torch.Tensor:
#         """
#         Args:
#             input: input features,
#                 expected shapes ``BxF`` where ``B``
#                 is batch dimension and ``F`` is an
#                 input feature dimension.
#             factor: disguising factor
#                 expected shapes ``Bx1`` where
#                 ``B`` is batch dimension.
#             label: target classes,
#                 expected shapes ``B`` where
#                 ``B`` is batch dimension.

#         Returns:
#             tensor (logits) with shapes ``BxC``
#             where ``C`` is a number of classes.
#         """
#         cos_theta = torch.bmm(
#             F.normalize(input).unsqueeze(0).expand(self.k, *input.shape),  # k*b*f
#             F.normalize(self.weight, dim=1),  # normalize in_features dim  # k*f*c
#         )  # k*b*c    
#         cos_theta, index = torch.max(cos_theta, dim=0)  # b*c 分别取与不同说话人K个sub-center中最相似的那个.
#         theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))
#         one_hot   = torch.zeros_like(cos_theta)
#         index_spk = label.view(-1, 1).long()
#         one_hot.scatter_(1, index_spk, 1) #one_hot label
#         selected  = torch.where(
#             theta > self.threshold, torch.zeros_like(one_hot), one_hot
#         )
#         logits    = torch.cos(torch.where(selected.bool(), theta + self.m, theta))    
#         logits   *= self.s
#         cls_loss  = self.loss_fn(logits, label)
        
#         prec1     = accuracy(logits.detach(), label.detach(), topk=(1,))[0]
        
#         batch = input.shape[0]
#         index_b = torch.tensor(range(batch)).unsqueeze(-1).long()
#         index_k = index[index_b, index_spk].long()
#         sub_center = self.weight.unsqueeze(1).expand(self.k, batch, self.in_features, self.out_features)[index_k, index_b, :, index_spk]                               
#         cc_center = torch.mean(self.weight,dim=0).unsqueeze(0).expand(batch, self.in_features, self.out_features)[index_b, :, index_spk]   
#         cos_theta_n = torch.bmm(
#             F.normalize(sub_center, dim=-1),     # b*1*c
#             F.normalize(cc_center,  dim=-1).mT,    # b*c*1
#         ).squeeze(-1)  # b*1
        
#         theta_n = torch.acos(torch.clamp(cos_theta_n, -1.0 + self.eps, 1.0 - self.eps))
#         pred_a   = 12*torch.tanh(self.fw(theta_n)) 
#         pred_loss = self.smoothl1_fn(pred_a, factor)       
        
       
#         return cls_loss, pred_loss, prec1 #, logits
    



############### aamsoftmaxv ###################
class DAMSoftmax(nn.Module):
    """Implementation of
    `Sub-center ArcFace: Boosting Face Recognition
    by Large-scale Noisy Web Faces`_.

    .. _Sub-center ArcFace\: Boosting Face Recognition \
        by Large-scale Noisy Web Faces:
        https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature,
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.
        k: number of possible class centroids.
            Default: ``3``.
        eps (float, optional): operation accuracy.
            Default: ``1e-6``.

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.

    Example:
        >>> layer = SubCenterArcFace(5, 10, s=1.31, m=0.35, k=2)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """

    def __init__(  # noqa: D107
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.5,
        c: float = 1.5, # f_a(a)=c^(a/12)
        k: int = 128, # disguising factor: [-8, 8] defualt = 17(model0000)， 128(model0020)
        eps: float = 1e-6,
    ):
        super(DAMSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        self.c = c
        self.k = k
        self.eps = eps
        
        self.weight = nn.Parameter(torch.FloatTensor(k, in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.mse_fn  = nn.MSELoss()
        self.softmax_fn = nn.Softmax()
        
        # self.pred_head  = nn.Sequential(nn.Linear(out_features, 256),
        #                                 nn.Linear(256, 1))
        # for layer in self.pred_head:
        #     nn.init.xavier_uniform_(layer.weight)
        self.threshold  = math.pi - self.m

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "DAMSoftmax("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"m={self.m},"
            f"c={self.c},"
            f"k={self.k},"
            f"eps={self.eps},"
            f"loss_fn={self.loss_fn}"
            ")"
        )
        return rep

    
    def forward(self, input: torch.Tensor, factor: torch.Tensor, label: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            factor: disguising factor
                expected shapes ``Bx1`` where
                ``B`` is batch dimension.
            label: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.

        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes.
        """
        func_a =  torch.pow(self.c, (factor/12))*self.m
        threshold  = math.pi - func_a
        # a = [3,4,5]; (1, a): (1, [3,4,5]); (1, *a): (1, 3, 4, 5)
        # .unsqueeze(0).expand(n, ...) = expand(n, ...)
#         cos_theta = torch.bmm(
#             F.normalize(input).unsqueeze(0).expand(self.k, *input.shape),  # k*b*f
#             F.normalize(self.weight, dim=1),  # normalize in_features dim  # k*f*c
#         )  # k*b*c
#         # selected the most similar sub-center
#         cos_theta = torch.max(cos_theta, dim=0)[0]  # b*c
#         # make the cos_theta in (-1, 1)
#         theta     = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))
#         # make zero vector
#         one_hot   = torch.zeros_like(cos_theta)
#         # A.scatter_(r/c, index, ele): replace the elements in A with 'ele'; in the index (0: row; c: column)
#         # label = [A, B, C, ... ,N]; label.view(-1, 1) = label.squeeze(0).T
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1) #one_hot label
#         # 与cluster-center夹角超过threshold, 则认为不属于这类, label被放弃, 否则都属于这类
#         # torch.where(cond, a, b); if cond==True, replace b with a 
#         selected  = torch.where(
#             theta > threshold, torch.zeros_like(one_hot), one_hot
#         )
#         # theta是与每个cluster-center之间的夹角, c表示多少个类别, 即多少个夹角
#         # logits    = torch.cos(torch.where(selected.bool(), theta + func_a, theta)) 
#         logits    = torch.cos(torch.where(selected.bool(), theta + self.m, theta))    
#         logits   *= self.s
#         loss      = self.loss_fn(logits, label)
                
#         prec1     = accuracy(logits.detach(), label.detach(), topk=(1,))[0]
#         return loss, prec1

        # func_a = torch.pow(self.c, (factor/12))*self.m
        # threshold = math.pi - func_a
        cos_theta = torch.bmm(
            F.normalize(input).unsqueeze(0).expand(self.k, *input.shape),  # k*b*f
            F.normalize(self.weight, dim=1),  # normalize in_features dim  # k*f*c
        )  # k*b*c

        cos_theta = torch.max(cos_theta, dim=0)[0]  # b*c 分别取与不同说话人K个sub-center中最相似的那个.
        theta     = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))
        # cos_theta_cc = torch.mean(cos_theta, dim=0)[0]  # b*c
        # theta_cc     = torch.acos(torch.clamp(cos_theta_cc, -1.0 + self.eps, 1.0 - self.eps))
        # theta_n      = theta - theta_cc # b*c
       
        one_hot   = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1) #one_hot label
        selected  = torch.where(
            theta > threshold, torch.zeros_like(one_hot), one_hot
        )

        logits    = torch.cos(torch.where(selected.bool(), theta + func_a, theta))    
        logits   *= self.s
        cls_loss  = self.loss_fn(logits, label)
        
        # pred_a    = 12*torch.tanh(self.pred_head(theta_n)) # model:0002     
        # pred_a   = 12*torch.log2((1.5*torch.tanh(self.pred_head(theta_n))+2.5)/2) # model:0008
        # pred_loss = self.mse_fn(pred_a, factor)       
        
        prec1     = accuracy(logits.detach(), label.detach(), topk=(1,))[0]
        return cls_loss, prec1 #, logits
    
    
    def forward_(self, input: torch.Tensor):
        cos_theta_all = torch.bmm(
            F.normalize(input).unsqueeze(0).expand(self.k, *input.shape),  # k*b*f
            F.normalize(self.weight, dim=1),  # normalize in_features dim   # k*f*c
        )  # k*b*c       
        
        cos_theta_ps = torch.max(cos_theta_all, dim=0)[0]  # b*c
        cos_theta_cc = torch.mean(cos_theta_all, dim=0)  # b*c
        theta_ps     = torch.acos(torch.clamp(cos_theta_ps, -1.0 + self.eps, 1.0 - self.eps))
        theta_cc     = torch.acos(torch.clamp(cos_theta_cc, -1.0 + self.eps, 1.0 - self.eps))
        theta_n      = theta_ps - theta_cc # b*c
 
        return theta_n # b*c*k ——> b*a





# ############## damsoftmax ##################
# class DAMSoftmax(nn.Module):
#     def __init__(  # noqa: D107
#         self,
#         in_features: int,
#         out_features: int,
#         s: float = 64.0,
#         m: float = 0.5,
#         k: int = 128, 
#         eps: float = 1e-6,
#     ):
#         super(DAMSoftmax, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.k = k
#         self.eps = eps
        
#         self.weight = nn.Parameter(torch.FloatTensor(k, in_features, out_features))
#         nn.init.xavier_uniform_(self.weight)
        
#         self.fw = nn.Linear(1, 1)
#         nn.init.xavier_uniform_(self.fw.weight)
        
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.mse_fn  = nn.MSELoss()
#         self.softmax_fn = nn.Softmax()
#         self.smoothl1_fn = nn.SmoothL1Loss()
#         self.threshold  = math.pi - self.m
        
        

#     def __repr__(self) -> str:
#         """Object representation."""
#         rep = (
#             "DAMSoftmax("
#             f"in_features={self.in_features},"
#             f"out_features={self.out_features},"
#             f"s={self.s},"
#             f"m={self.m},"
#             f"k={self.k},"
#             f"eps={self.eps},"
#             f"loss_fn={self.loss_fn}"
#             ")"
#         )
#         return rep

    
#     def forward(self, input: torch.Tensor, factor: torch.Tensor, label: torch.LongTensor) -> torch.Tensor:
#         """
#         Args:
#             input: input features,
#                 expected shapes ``BxF`` where ``B``
#                 is batch dimension and ``F`` is an
#                 input feature dimension.
#             factor: disguising factor
#                 expected shapes ``Bx1`` where
#                 ``B`` is batch dimension.
#             label: target classes,
#                 expected shapes ``B`` where
#                 ``B`` is batch dimension.

#         Returns:
#             tensor (logits) with shapes ``BxC``
#             where ``C`` is a number of classes.
#         """
        
#         b = factor//0.1
#         a = 0.1*b
#         batch = input.shape[0]
#         index_a = ((a+12)/0.1).long()
#         index_b = torch.tensor(range(batch)).unsqueeze(-1).long()
        
#         # sub_center = self.weight.unsqueeze(1).expand(self.k, batch, self.in_features, self.out_features)[index_a, index_b, :, :].squeeze(1) 
#         # cos_theta = torch.bmm(
#         #     F.normalize(input).unsqueeze(1),  # b*1*f
#         #     F.normalize(sub_center, dim=1),  # b*f*c
#         # ).squeeze(1)  # b*c
        
#         cos_theta = torch.bmm(
#             F.normalize(input).unsqueeze(0).expand(self.k, *input.shape),  # k*b*f
#             F.normalize(self.weight, dim=1),  # normalize in_features dim  # k*f*c
#         )[index_a, index_b, :].squeeze(1)  # b*c
        
#         theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))
#         one_hot   = torch.zeros_like(cos_theta)
#         index_spk = label.view(-1, 1).long()
#         one_hot.scatter_(1, index_spk, 1) #one_hot label
#         selected  = torch.where(
#             theta > self.threshold, torch.zeros_like(one_hot), one_hot
#         )
#         logits    = torch.cos(torch.where(selected.bool(), theta + self.m, theta))    
#         logits   *= self.s
#         cls_loss  = self.loss_fn(logits, label)
        
#         prec1     = accuracy(logits.detach(), label.detach(), topk=(1,))[0]
       
#         return cls_loss, prec1 #, logits
  




# ############## damsoftmax ##################
# class DAMSoftmax(nn.Module):
#     def __init__(  # noqa: D107
#         self,
#         in_features: int,
#         out_features: int,
#         s: float = 64.0,
#         m: float = 0.5,
#         k1: int = 25, 
#         k2: int = 5, 
#         eps: float = 1e-6,
#     ):
#         super(DAMSoftmax, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.k1 = k1
#         self.k2 = k2
#         self.eps = eps
        
#         self.weight = nn.Parameter(torch.FloatTensor(k2, k1, in_features, out_features))
#         nn.init.xavier_uniform_(self.weight)
        
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.mse_fn  = nn.MSELoss()
#         self.softmax_fn = nn.Softmax()
#         self.smoothl1_fn = nn.SmoothL1Loss()
#         self.threshold  = math.pi - self.m
        
        

#     def __repr__(self) -> str:
#         """Object representation."""
#         rep = (
#             "DAMSoftmax("
#             f"in_features={self.in_features},"
#             f"out_features={self.out_features},"
#             f"s={self.s},"
#             f"m={self.m},"
#             f"k={self.k},"
#             f"eps={self.eps},"
#             f"loss_fn={self.loss_fn}"
#             ")"
#         )
#         return rep

    
#     def forward(self, input: torch.Tensor, factor: torch.Tensor, label: torch.LongTensor) -> torch.Tensor:
#         """
#         Args:
#             input: input features,
#                 expected shapes ``BxF`` where ``B``
#                 is batch dimension and ``F`` is an
#                 input feature dimension.
#             factor: disguising factor
#                 expected shapes ``Bx1`` where
#                 ``B`` is batch dimension.
#             label: target classes,
#                 expected shapes ``B`` where
#                 ``B`` is batch dimension.

#         Returns:
#             tensor (logits) with shapes ``BxC``
#             where ``C`` is a number of classes.
#         """
        
#         b = factor//0.1
#         a = 0.1*b
#         batch = input.shape[0]
#         index_a = ((a+12)/0.1).long()
#         index_b = torch.tensor(range(batch)).unsqueeze(-1).long()
        
#         cos_theta = torch.bmm(
#             F.normalize(input).unsqueeze(0).unsqueeze(0).expand(self.k2, self.k1, *input.shape),  # k2*k1*b*f
#             F.normalize(self.weight, dim=2),  # normalize in_features dim  # k2*k1*f*c
#         )[index_a, index_b, :].squeeze(1)  # b*c
        
#         theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))
#         one_hot   = torch.zeros_like(cos_theta)
#         index_spk = label.view(-1, 1).long()
#         one_hot.scatter_(1, index_spk, 1) #one_hot label
#         selected  = torch.where(
#             theta > self.threshold, torch.zeros_like(one_hot), one_hot
#         )
#         logits    = torch.cos(torch.where(selected.bool(), theta + self.m, theta))    
#         logits   *= self.s
#         cls_loss  = self.loss_fn(logits, label)
        
#         prec1     = accuracy(logits.detach(), label.detach(), topk=(1,))[0]
       
#         return cls_loss, prec1 #, logits
    


    

    

    

# class DAMSoftmax(nn.Module):
#     """Implementation of
#     `Sub-center ArcFace: Boosting Face Recognition
#     by Large-scale Noisy Web Faces`_.

#     .. _Sub-center ArcFace\: Boosting Face Recognition \
#         by Large-scale Noisy Web Faces:
#         https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf

#     Args:
#         in_features: size of each input sample.
#         out_features: size of each output sample.
#         s: norm of input feature,
#             Default: ``64.0``.
#         m: margin.
#             Default: ``0.5``.
#         k: number of possible class centroids.
#             Default: ``3``.
#         eps (float, optional): operation accuracy.
#             Default: ``1e-6``.

#     Shape:
#         - Input: :math:`(batch, H_{in})` where
#           :math:`H_{in} = in\_features`.
#         - Output: :math:`(batch, H_{out})` where
#           :math:`H_{out} = out\_features`.

#     Example:
#         >>> layer = SubCenterArcFace(5, 10, s=1.31, m=0.35, k=2)
#         >>> loss_fn = nn.CrossEntropyLoss()
#         >>> embedding = torch.randn(3, 5, requires_grad=True)
#         >>> target = torch.empty(3, dtype=torch.long).random_(10)
#         >>> output = layer(embedding, target)
#         >>> loss = loss_fn(output, target)
#         >>> loss.backward()

#     """

#     def __init__(  # noqa: D107
#         self,
#         in_features: int,
#         out_features: int,
#         s: float = 64.0,
#         m: float = 0.5,
#         c: float = 1.5, # f_a(a)=c^(a/12)
#         k: int = 128, # disguising factor: [-8, 8] defualt = 17(model0000)， 128(model0020)
#         eps: float = 1e-6,
#     ):
#         super(DAMSoftmax, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         self.s = s
#         self.m = m
#         self.c = c
#         self.k = k
#         self.eps = eps
        
#         self.center = nn.Parameter(torch.FloatTensor(k, in_features, out_features))
#         nn.init.xavier_uniform_(self.center)
        
#         self.fw = nn.Linear(out_features, 1)
#         nn.init.xavier_uniform_(self.fw.weight)
        
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.mse_fn  = nn.MSELoss()
#         self.l1_fn   = nn.SmoothL1Loss()
#         self.softmax_fn = nn.Softmax()
        
#         # self.threshold  = math.pi - self.m

#     def __repr__(self) -> str:
#         """Object representation."""
#         rep = (
#             "DAMSoftmax("
#             f"in_features={self.in_features},"
#             f"out_features={self.out_features},"
#             f"s={self.s},"
#             f"m={self.m},"
#             f"c={self.c},"
#             f"k={self.k},"
#             f"eps={self.eps},"
#             f"loss_fn={self.loss_fn}"
#             ")"
#         )
#         return rep

    
#     def forward(self, input: torch.Tensor, factor: torch.Tensor, label: torch.LongTensor) -> torch.Tensor:
#         """
#         Args:
#             input: input features,
#                 expected shapes ``BxF`` where ``B``
#                 is batch dimension and ``F`` is an
#                 input feature dimension.
#             factor: disguising factor
#                 expected shapes ``Bx1`` where
#                 ``B`` is batch dimension.
#             label: target classes,
#                 expected shapes ``B`` where
#                 ``B`` is batch dimension.

#         Returns:
#             tensor (logits) with shapes ``BxC``
#             where ``C`` is a number of classes.
#         """
#         func_a     = torch.pow(self.c, (factor/12))*self.m
#         threshold  = math.pi - func_a        
#         cos_theta_all = torch.bmm(
#             F.normalize(input).unsqueeze(0).expand(self.k, *input.shape),  # k*b*f
#             F.normalize(self.center, dim=1),  # normalize in_features dim   # k*f*c
#         )  # k*b*c       
#         cos_theta_ps = torch.max(cos_theta_all,  dim=0)[0]  # b*c
#         cos_theta_cc = torch.mean(cos_theta_all, dim=0)    # b*c
        
#         theta_ps = torch.acos(torch.clamp(cos_theta_ps,    -1.0 + self.eps, 1.0 - self.eps))
#         theta_cc = torch.acos(torch.clamp(cos_theta_cc, -1.0 + self.eps, 1.0 - self.eps))
#         theta_n  = theta_ps - theta_cc # b*c
       
#         one_hot   = torch.zeros_like(cos_theta_ps)
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1) #one_hot label
#         selected  = torch.where(
#             theta_ps > threshold, torch.zeros_like(one_hot), one_hot
#         )

#         logits    = torch.cos(torch.where(selected.bool(), theta_ps + func_a, theta_ps))    
#         logits   *= self.s
#         cls_loss  = self.loss_fn(logits, label)
   
#         pred_a    = 12*torch.tanh(self.fw(theta_n))  
#         # pred_a    = 12*torch.log2((1.5*torch.tanh(self.fw(theta_n))+2.5)/2)  
#         pred_loss = self.l1_fn(pred_a, factor)
#         prec1     = accuracy(logits.detach(), label.detach(), topk=(1,))[0]
        
#         return cls_loss, pred_loss, prec1 #, logits
    
    

    
class PredictionLoss(nn.Module):
    def __init__(
                 self,
                 in_features: int,
                 hidden_features: int,
                 k: int = 1299,         #spk_nums
                 eps: float = 1e-6,
                 model: str = 'train'
                 ):
        super(PredictionLoss, self).__init__()
        self.in_features     = in_features
        self.hidden_features = hidden_features
        self.centers = nn.Parameter(torch.FloatTensor(k, in_features, 1)) # k*f: cluster-center
        nn.init.xavier_uniform_(self.centers)   

        self.k = k
        self.eps = eps
        if model=='eval':
            self.loss = nn.L1Loss()          
        else:
            self.loss = nn.SmoothL1Loss()

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "PredictionLoss("
            f"in_features     = {self.in_features},"
            f"hidden_features = {self.hidden_features},"
        )
        return rep

    def forward(self, input: torch.Tensor, factor: torch.Tensor, label: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            factor: disguising factor
                expected shapes ``Bx1`` where
                ``B`` is batch dimension.
                
        Returns:
            tensor with shapes ``B``
        """
        batch = input.shape[0]
        # Label: b
        index_spk = label.view(-1, 1).long()
        index_b = torch.tensor(range(batch)).unsqueeze(-1).long()
        
        pred_alpha = torch.bmm(
            input.unsqueeze(0).expand(self.k, *input.shape),  # k*b*c 
            self.centers,                                     # k*c*1
        ) [index_spk, index_b, :].squeeze(1) # b*1                              
  
        predict_a = 12*torch.tanh(pred_alpha) 
        loss = self.loss(predict_a, factor)

        return loss, predict_a
    

    


    
# class PredictionLoss(nn.Module):
#     def __init__(
#                  self,
#                  in_features: int,
#                  hidden_features: int,
#                  k: int = 256,       #spk_nums
#                  eps: float = 1e-6,
#                  model: str = 'train'
#                  ):
#         super(PredictionLoss, self).__init__()
#         self.in_features     = in_features
#         self.hidden_features = hidden_features
        
        
#         self.weights = nn.Parameter(torch.FloatTensor(k, in_features, hidden_features)) # k*f*128
#         nn.init.xavier_uniform_(self.weights)

#         self.attn = nn.MultiheadAttention(768, 3)
#         # nn.init.xavier_uniform_(self.attn.weight)
#         self.mlp = nn.Linear(hidden_features, 1)
#         nn.init.xavier_uniform_(self.mlp.weight)                     

#         self.k = k
#         self.eps = eps
#         if model=='eval':
#             self.loss = nn.L1Loss()          
#         else:
#             self.loss = nn.SmoothL1Loss()

#     def __repr__(self) -> str:
#         """Object representation."""
#         rep = (
#             "PredictionLoss("
#             f"in_features     = {self.in_features},"
#             f"hidden_features = {self.hidden_features},"
#         )
#         return rep

#     def forward(self, input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             input: input features,
#                 expected shapes ``BxF`` where ``B``
#                 is batch dimension and ``F`` is an
#                 input feature dimension.
#             factor: disguising factor
#                 expected shapes ``Bx1`` where
#                 ``B`` is batch dimension.
                
#         Returns:
#             tensor with shapes ``B``
#         """
#         cos_theta = torch.bmm(
#             F.normalize(input).unsqueeze(0).expand(self.k, *input.shape),  # k*b*f 999*32*768
#             F.normalize(self.weights, dim=1),  # normalize in_features dim   # k*f*c 999*768*17
#         )  # 999*32*17  
#         cos_theta = torch.max(cos_theta, dim=0)[0]  # b*c
#         theta     = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))
#         # x = torch.cat([input.unsqueeze(1), cc.unsqueeze(1)], dim=1)
#         # x = self.layer1(x)
#         # x = self.flat(x)
#         # x, _ = self.attn(input, input, input)
#         x    = self.mlp(theta)
        
#         # predict_a = 8*torch.tanh(self.Predictionhead(input)) # model:0002
#         # predict_a = 8*torch.tanh(self.layer1(input)) # model:0004
#         # predict_a = 8*torch.tanh(x) 
#         # predict_a = 12*torch.log2(torch.tanh(self.Predictionhead(input))+self.eps) # model:0006
#         predict_a = 12*torch.tanh(x) 
#         # predict_a = 12*torch.log2((1.5*torch.tanh(x)+2.5)/2) # model:0008
#         loss = self.loss(predict_a, factor)

#         return loss, predict_a
    
    
    
    

# class PredictionLoss(nn.Module):
#     def __init__(
#                  self,
#                  in_features: int,
#                  hidden_features: int,
#                  s = 64,
#                  k: int = 1299,       #spk_nums
#                  m = 0.15,
#                  eps: float = 1e-6,
#                  ):
#         super(PredictionLoss, self).__init__()
#         self.in_features     = in_features
#         self.hidden_features = hidden_features  
#         self.weights = nn.Parameter(torch.FloatTensor(k, in_features, hidden_features)) # k*f*256
#         nn.init.xavier_uniform_(self.weights)

#         self.s = s
#         self.m = m
#         self.k = k
#         self.eps = eps
#         self.threshold = math.pi - self.m
#         self.loss_fn = nn.CrossEntropyLoss()
    

#     def __repr__(self) -> str:
#         """Object representation."""
#         rep = (
#             "PredictionLoss("
#             f"in_features     = {self.in_features},"
#             f"hidden_features = {self.hidden_features},"
#         )
#         return rep

#     def forward(self, input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             input: input features,
#                 expected shapes ``BxF`` where ``B``
#                 is batch dimension and ``F`` is an
#                 input feature dimension.
#             factor: disguising factor
#                 expected shapes ``Bx1`` where
#                 ``B`` is batch dimension.
                
#         Returns:
#             tensor with shapes ``B``
#         """
#         b = factor//0.1
#         a = 0.1*b
#         label = ((a+12)/0.1).long()
        
#         cos_theta = torch.bmm(
#             F.normalize(input).unsqueeze(0).expand(self.k, *input.shape),    # k*b*f 1299*b*768
#             F.normalize(self.weights, dim=1),  # normalize in_features dim   # k*f*c 1299*768*256
#         )  # 1299*b*256  
#         cos_theta = torch.max(cos_theta, dim=0)[0]  # b*c
#         theta     = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)) 
#         one_hot   = torch.zeros_like(cos_theta)
#         one_hot.scatter_(1, label, 1) #one_hot label
#         selected  = torch.where(
#             theta > self.threshold, torch.zeros_like(one_hot), one_hot
#         )
#         logits    = torch.cos(torch.where(selected.bool(), theta + self.m, theta))    
#         logits   *= self.s
        
#         cls_loss  = self.loss_fn(logits, label.squeeze(-1))
        
#         prec2     = accuracy(logits.detach(), label.detach(), topk=(1,))[0]     
#         predict_a = (label*0.1-12).float()
        
#         return cls_loss, prec2, predict_a
    
    
    
    
    
    
    
def weights_nonzero_speech(target):
    # target : B x T x mel
    # Assign weight 1.0 to all labels except for padding (id=0).
    dim = target.size(-1)
    return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)


def l1_loss(decoder_output, target):
    # decoder_output : B x T x n_mel
    # target : B x T x n_mel
    l1_loss = F.l1_loss(decoder_output, target, reduction="none")
    weights = weights_nonzero_speech(target)
    l1_loss = (l1_loss * weights).sum() / weights.sum()
    return l1_loss


def get_mel_loss(mel_predictions, mel_targets):
    mel_targets.requires_grad = False
    mel_loss = l1_loss(mel_predictions, mel_targets)
    return mel_loss


def jcu_loss_fn(logit_cond, logit_uncond, label_fn):
    n_scale = len(logit_uncond)
    uncond_loss = []
    # conditional loss
    cond_loss = F.binary_cross_entropy_with_logits(logit_cond[-1], label_fn(logit_cond[-1]))
    # unconditional loss
    for i in range(n_scale):
        loss = F.binary_cross_entropy_with_logits(logit_uncond[i][-1], label_fn(logit_uncond[i][-1]))
        uncond_loss.append(loss)     
        
    return 0.5 * (cond_loss + sum(uncond_loss)/n_scale)


def discriminator_loss(disc_real_cond, disc_real_uncond, disc_fake_cond, disc_fake_uncond):
    r_loss = jcu_loss_fn(disc_real_cond, disc_real_uncond, torch.ones_like)
    g_loss = jcu_loss_fn(disc_fake_cond, disc_fake_uncond, torch.zeros_like)
    return 0.5 * (r_loss + g_loss)



def generator_loss(disc_fake_cond, disc_fake_uncond):
    g_loss = jcu_loss_fn(disc_fake_cond, disc_fake_uncond, torch.ones_like)
    return g_loss


def get_fm_loss(disc_real_cond, disc_real_uncond, disc_fake_cond, disc_fake_uncond):
    n_scale    = len(disc_real_uncond)
    n_layers   = 5
    loss_fm_c  = 0
    loss_fm_uc = []
    loss_uc = 0
    feat_weights = 4.0 / (n_layers + 1)
    # conditional loss
    for j in range(n_layers):
        loss_fm_c += F.l1_loss(disc_fake_cond[j], disc_real_cond[j].detach())
         
    # unconditional loss
    for i in range(n_scale):
        for j in range(n_layers):
            loss_uc += F.l1_loss(disc_fake_uncond[i][j], disc_real_uncond[i][j].detach())       
        loss_fm_uc.append(loss_uc)   
    
    return feat_weights * 0.5 *(loss_fm_c + sum(loss_fm_uc)/n_scale)
