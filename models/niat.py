import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.transformers_encoder.transformer import TransformerEncoder


################################## FUSION MODULE ########################################
# args.feature_dims & args.hidden_dims & args.text_out & args.fusion_dim & args.fusion_dropouts & args.language & args.use_bert_finetune
class transformer_based(nn.Module):
    def __init__(self, args):
        super(transformer_based, self).__init__()
        self.args = args
        # BERT SUBNET FOR TEXT
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_bert_finetune)
        args.fusion_dim = args.fus_d_l+args.fus_d_a+args.fus_d_v
        orig_d_l, orig_d_a, orig_d_v = args.feature_dims
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(orig_d_l, args.fus_d_l, kernel_size=args.fus_conv1d_kernel_l, padding=(args.fus_conv1d_kernel_l-1)//2, bias=False)
        self.proj_a = nn.Conv1d(orig_d_a, args.fus_d_a, kernel_size=args.fus_conv1d_kernel_a, padding=(args.fus_conv1d_kernel_a-1)//2, bias=False)
        self.proj_v = nn.Conv1d(orig_d_v, args.fus_d_v, kernel_size=args.fus_conv1d_kernel_v, padding=(args.fus_conv1d_kernel_v-1)//2, bias=False)

        self.fusion_trans = TransformerEncoder(embed_dim=args.fus_d_l+args.fus_d_a+args.fus_d_v, num_heads=args.fus_nheads, layers=args.fus_layers, 
                                                attn_dropout=args.fus_attn_dropout, relu_dropout=args.fus_relu_dropout, res_dropout=args.fus_res_dropout,
                                                embed_dropout=args.fus_embed_dropout, attn_mask=args.fus_attn_mask)

    def forward(self, text_x, audio_x, video_x):
        x_l = self.text_model(text_x).transpose(1, 2)
        x_a = audio_x.transpose(1, 2) # batch_size, da, seq_len
        x_v = video_x.transpose(1, 2)
        
        proj_x_l = self.proj_l(x_l).permute(2, 0, 1) # seq_len, batch_size, dl
        proj_x_a = self.proj_a(x_a).permute(2, 0, 1)
        proj_x_v = self.proj_v(x_v).permute(2, 0, 1)

        trans_seq = self.fusion_trans(torch.cat((proj_x_l, proj_x_a, proj_x_v), axis=2))
        if type(trans_seq) == tuple:
            trans_seq = trans_seq[0]

        return trans_seq[0] # Utilize the [CLS] of text for full sequences representation.    

FUSION_MODULE_MAP = {
   'structure_one': transformer_based,
}

class FUSION(nn.Module):
    """ 将三模态对齐序列融合为一个特征向量 
    """
    def __init__(self, args):
        super(FUSION, self).__init__()

        lastModel = FUSION_MODULE_MAP[args.fusion]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x):
        """ 输入三模态对齐数据，输出融合多模态特征向量。
            text_x: [batch_size, 3, seq_len] (bert_token);
            audio_x: [batch_size, seq_len, da]; 
            video_x: [batch_size, seq_len, dv]
        """
        return self.Model(text_x, audio_x, video_x)

################################## RECONSTRUCTION MODULE ########################################
# args.fusion_dim & args.rec_hidden_dim1 & args.rec_dropout & args.rec_hidden_dim2
class decoder_v1(nn.Module):
    """效仿ARGF模型"""
    def __init__(self, args):
        super(decoder_v1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.fusion_dim, args.rec_hidden_dim1),
            nn.Dropout(args.rec_dropout),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.rec_hidden_dim1, args.rec_hidden_dim2),
            nn.Dropout(args.rec_dropout),
            nn.BatchNorm1d(args.rec_hidden_dim2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.rec_hidden_dim2, args.fusion_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
    
REC_MODULE_MAP = {
   'structure_one': decoder_v1,
}

class RECONSTRUCTION(nn.Module):
    """ 将缺失数据源得到的融合特征重构为完整数据的融合特征
    """
    def __init__(self, args):
        super(RECONSTRUCTION, self).__init__()

        lastModel = REC_MODULE_MAP[args.reconstruction]
        self.Model = lastModel(args)

    def forward(self, fusion_feature):
        """ 输入缺失数据源得到的融合特征向量 [batch_size, d_fusion]
            输出重构的融合特征向量 [batch_size, d_fusion]
        """
        return self.Model(fusion_feature)

################################## DISCRIMINATOR MODULE ########################################
# args.fusion_dim & args.disc_hidden_dim1 & args.disc_hidden_dim2
class disc_two_class(nn.Module):
    """效仿ARGF模型"""
    def __init__(self, args):
        """ Basic Binary Discriminator. 
        """
        super(disc_two_class, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm1d(args.fusion_dim),
            nn.Linear(args.fusion_dim, args.disc_hidden_dim1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.disc_hidden_dim1, args.disc_hidden_dim2),
            nn.Tanh(),
            nn.Linear(args.disc_hidden_dim2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

DISC_MODULE_MAP = {
   'structure_one': disc_two_class,
}

class DISCRIMINATOR(nn.Module):
    """ 将三模态对齐序列融合为一个特征向量 
    """
    def __init__(self, args):
        super(DISCRIMINATOR, self).__init__()

        lastModel = DISC_MODULE_MAP[args.discriminator]
        self.Model = lastModel(args)


    def forward(self, fusion_feature):
        """ 输入 （完整/数据源1/数据源2 …） 的融合特征向量 [batch_size, d_fusion]
            输出 数据源分类结果 |数据源种类 + 1|
        """
        return self.Model(fusion_feature)

################################## CLASSIFIER MODULE ########################################
# args.fusion_dim & args.clf_dropout & args.clf_hidden_dim
class classifier_v1(nn.Module):

    def __init__(self, args):

        super(classifier_v1, self).__init__()
        self.norm = nn.BatchNorm1d(args.fusion_dim)
        self.drop = nn.Dropout(args.clf_dropout)
        self.linear_1 = nn.Linear(args.fusion_dim, args.clf_hidden_dim)
        self.linear_2 = nn.Linear(args.clf_hidden_dim, 1)
        # self.linear_3 = nn.Linear(hidden_size, hidden_size)
       
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, fusion_feature):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(fusion_feature)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        # y_2 = F.sigmoid(self.linear_2(y_1))
        y_2 = torch.sigmoid(self.linear_2(y_1))
        # 强制将输出结果转化为 [-3,3] 之间
        output = y_2 * self.output_range + self.output_shift

        return output

CLF_MODULE_MAP = {
   'structure_one': classifier_v1
}

class CLASSIFIER(nn.Module):
    """ 将三模态对齐序列融合为一个特征向量 
    """
    def __init__(self, args):
        super(CLASSIFIER, self).__init__()

        lastModel = CLF_MODULE_MAP[args.classifier]
        self.Model = lastModel(args)

    def forward(self, fusion_feature):
        """ 输入 （完整/数据源1/数据源2 …） 的融合特征向量 [batch_size, d_fusion]
            输出 情感极性回归值 [batch_size, 1]
        """

        return self.Model(fusion_feature)

################################## OVERALL MODEL ARCHITECTURE ########################################

class niat(nn.Module):
   
    def __init__(self, args):
        super(niat, self).__init__()

        self.args = args
        self.fusion = FUSION(args)

        self.reconstruction = RECONSTRUCTION(args)
        self.discriminator = DISCRIMINATOR(args)

        self.classifier = CLASSIFIER(args)
        

    def forward(self, text_x, audio_x, video_x):
        pass