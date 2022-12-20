"""
AIO -- All Model in One
"""
import torch.nn as nn
from .niat import niat
from .niat_ablation import niat_wo_da, niat_wo_dis, niat_wo_rec, niat_wo_dis_rec
__all__ = ['AMIO']

MODEL_MAP = {
    'niat': niat,
    'niat_wo_da': niat_wo_da,
    'niat_wo_dis': niat_wo_dis,
    'niat_wo_rec': niat_wo_rec,
    'niat_wo_dis_rec': niat_wo_dis_rec,
}

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        lastModel = MODEL_MAP[args.modelName]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x):
        return self.Model(text_x, audio_x, video_x)