import logging
from operator import mod
from numpy.core.fromnumeric import size
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import itertools
from torch.autograd import Variable

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

logger = logging.getLogger('MSA')

class niat_wo_rec():
    
    def __init__(self, args):
        self.args = args
        """ args.decay, args.learning_rate, args.KeyEval, args.alpha (rec_loss & discrimator_loss)
        """
        self.criterion = nn.L1Loss()
        self.adversarial_loss = nn.BCELoss()

        self.metrics = MetricsTop().getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        if self.args.use_bert_finetune:
            # OPTIMIZER: finetune Bert Parameters.
            bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            bert_params = list(model.Model.fusion.Model.text_model.named_parameters())

            bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
            bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
            model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]

            optimizer_grouped_parameters = [
                {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
                {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
                {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
            ]
            optimizer = optim.Adam(optimizer_grouped_parameters)

            optimizer_D = optim.Adam(model.Model.discriminator.parameters(), lr=self.args.learning_rate_other, weight_decay=self.args.weight_decay_other) # betas = (b1, b2)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.decay)

            optimizer_D = optim.Adam(model.Model.discriminator.parameters(), lr=self.args.learning_rate, weight_decay=self.args.decay) # betas = (b1, b2)


        # initilize results
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred, y_true = [], []
            model.train()
            # train_loss = 0.0
            avg_rloss = []
            avg_closs = []
            avg_dloss = []

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    # 目前提出的模型只能使用含有数据增强的数据。
                    vision_lm = batch_data['vision_lm'].to(self.args.device)
                    audio_lm = batch_data['audio_lm'].to(self.args.device)
                    text_lm = batch_data['text_lm'].to(self.args.device)

                    # encoder-decoder
                    optimizer.zero_grad()

                    fusion_feature_x = model.Model.fusion(text, audio, vision)
                    fusion_feature_lm = model.Model.fusion(text_lm, audio_lm, vision_lm)
                    # recon_fusion_f = model.Model.reconstruction(fusion_feature_lm)

                    # rl1 = self.pixelwise_loss(fusion_feature_x, recon_fusion_f)
                    # avg_rloss.append(rl1.item())
                    
                    # Add Label smoothing Strategy.
                    # real = (torch.rand(logits_real.size()) * 0.25 + 0.85).clone().detach().to(device)
                    # fake = (torch.rand(logits_fake.size()) * 0.15).clone().detach().to(device)
                    valid = torch.ones(size=[labels.shape[0], 1], requires_grad=False).type_as(audio).to(self.args.device)
                    fake = torch.zeros(size=[labels.shape[0], 1], requires_grad=False).type_as(audio).to(self.args.device)

                    t = model.Model.discriminator(fusion_feature_lm)
                    advl1 = self.adversarial_loss(t, valid)
                    g_loss = self.args.alpha * (advl1)
                    g_loss.backward(retain_graph=True)

                    optimizer_D.zero_grad()

                    output_x = model.Model.classifier(fusion_feature_x)
                    y_pred.append(output_x.cpu())
                    y_true.append(labels.cpu())
                    output_lm = model.Model.classifier(fusion_feature_lm)
                    y_pred.append(output_lm.cpu())
                    y_true.append(labels.cpu()) 
                    c_loss = (self.criterion(output_x, labels)+self.criterion(output_lm, labels) * self.args.beta) / (1 + self.args.beta)
                    avg_closs.append(c_loss.item())
                    
                    c_loss.backward()
                    
                    real_loss = self.adversarial_loss(model.Model.discriminator(fusion_feature_x.clone().detach()), valid)
                    fake_loss = self.adversarial_loss(model.Model.discriminator(fusion_feature_lm.clone().detach()), fake)
                    
                    d_loss = 0.5 * (real_loss + fake_loss)
                    avg_dloss.append(d_loss.item())
                    d_loss.backward()

                    if self.args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], self.args.grad_clip)
                    optimizer.step()

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info("TRAIN-(%s) (%d/%d/%d)>> rloss: %.4f closs: %.4f dloss: %.4f %s" % (self.args.modelName, \
                        epochs - best_epoch, epochs, self.args.cur_time, np.mean(avg_rloss), np.mean(avg_closs), np.mean(avg_dloss), dict_to_str(train_results)))
            # validation
            val_results = self.do_valid(model, dataloader['valid'])
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return


    def do_valid(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    # 目前提出的模型只能使用含有数据增强的数据。
                    vision_lm = batch_data['vision_lm'].to(self.args.device)
                    audio_lm = batch_data['audio_lm'].to(self.args.device)
                    text_lm = batch_data['text_lm'].to(self.args.device)
                    
                    fusion_feature_x = model.Model.fusion(text, audio, vision)
                    fusion_feature_lm = model.Model.fusion(text_lm, audio_lm, vision_lm)

                    output_x = model.Model.classifier(fusion_feature_x)
                    loss = self.criterion(output_x, labels)
                    y_pred.append(output_x.cpu())
                    y_true.append(labels.cpu())
                    output_lm = model.Model.classifier(fusion_feature_lm)
                    loss += self.criterion(output_lm, labels)
                    y_pred.append(output_lm.cpu())
                    y_true.append(labels.cpu()) 

                    eval_loss += loss.item()

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logger.info("%s-(%s) >> %s" % (mode, self.args.modelName + '-' + self.args.augment, dict_to_str(eval_results)))
        return eval_results