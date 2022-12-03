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
        k: int = 17, # disguising factor: [-8, 8]
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
        # self.threshold = math.pi - self.m

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
        ##############################################
        func_a = torch.pow(self.c, (factor/12))*self.m
        threshold = math.pi - func_a
        ##############################################
        cos_theta = torch.bmm(
            F.normalize(input).unsqueeze(0).expand(self.k, *input.shape),  # k*b*f
            F.normalize(self.weight, dim=1),  # normalize in_features dim   # k*f*c
        )  # k*b*c
        cos_theta = torch.max(cos_theta, dim=0)[0]  # b*c
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        selected = torch.where(
            theta > threshold, torch.zeros_like(one_hot), one_hot
        )
  
        logits = torch.cos(torch.where(selected.bool(), theta + func_a, theta))
        logits *= self.s
        loss = self.loss_fn(logits, label)
        
        prec1 = accuracy(logits.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
    
    
    
class PredictionLoss(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
    ):
        super(PredictionLoss, self).__init__()
        self.in_features     = in_features
        self.hidden_features = hidden_features

        self.Predictionhead = nn.Sequential(nn.Linear(in_features, hidden_features),
                                            nn.Linear(hidden_features, 1))
        nn.init.xavier_uniform_(self.Predictionhead.weight)
        self.loss = nn.MSELoss()
        
    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "PredictionLoss("
            f"in_features     = {self.in_features},"
            f"hidden_features = {self.hidden_features},"
        )
        return rep

    def forward(self, input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
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
        predict_a = self.Predictionhead(input)
        loss = self.loss(predict_a, factor)

        return loss, predict_a
    
    
    
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
    # mel_predictions = mel_predictions.masked_fill(self.mel_masks_fill.unsqueeze(-1), 0)
    # mel_targets = mel_targets.masked_fill(self.mel_masks_fill.unsqueeze(-1), 0)
    mel_loss = l1_loss(mel_predictions, mel_targets)
    return mel_loss


def jcu_loss_fn(logit_cond, logit_uncond, label_fn):
    cond_loss = F.mse_loss(logit_cond, label_fn(logit_cond), reduction="mean")
    uncond_loss = F.mse_loss(logit_uncond, label_fn(logit_uncond), reduction="mean")
    return 0.5 * (cond_loss + uncond_loss)


def discriminator_loss(disc_real_cond, disc_real_uncond, disc_fake_cond, disc_fake_uncond):
    r_loss = jcu_loss_fn(disc_real_cond[-1], disc_real_uncond[-1], torch.ones_like)
    g_loss = jcu_loss_fn(disc_fake_cond[-1], disc_fake_uncond[-1], torch.zeros_like)
    return r_loss + g_loss


def generator_loss(disc_fake_cond, disc_fake_uncond):
    g_loss = jcu_loss_fn(disc_fake_cond[-1], disc_fake_uncond[-1], torch.ones_like)
    return g_loss


def get_fm_loss(disc_real_cond, disc_real_uncond, disc_fake_cond, disc_fake_uncond):
    loss_fm = 0
    n_layers = 3
    feat_weights = 4.0 / (n_layers + 1)
    for j in range(len(disc_fake_cond) - 1):
        loss_fm += feat_weights * 0.5 * (F.l1_loss(disc_real_cond[j].detach(), disc_fake_cond[j]) \
                                         + F.l1_loss(disc_real_uncond[j].detach(), disc_fake_uncond[j]))
    return loss_fm