"""
AIO -- All Trains in One
"""
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from utils.functions import list_to_str, dict_to_str, calculate_AUILC
from utils.metricsTop import MetricsTop
from .niat import *
from .niat_wo_da import *
from .niat_wo_dis import *
from .niat_wo_rec import *
from .niat_wo_dis_rec import *

__all__ = ['ATIO']

logger = logging.getLogger('MSA')


class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # single-task
            'niat': niat,
            'niat_wo_da': niat_wo_da,
            'niat_wo_dis': niat_wo_dis,
            'niat_wo_rec': niat_wo_rec,
            'niat_wo_dis_rec': niat_wo_dis_rec,
        }

    @staticmethod
    def do_test(model, dataloaders, args):
        loss_function = nn.L1Loss()
        model.eval()
        results = {}
        eval_loss = 0.0
        for n, dataloader in enumerate(dataloaders):
            y_pred, y_true = [], []
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(args.device)
                        audio = batch_data['audio'].to(args.device)
                        text = batch_data['text'].to(args.device)
                        labels = batch_data['labels']['M'].to(args.device).view(-1, 1)

                        fusion_feature_x = model.Model.fusion(text, audio, vision)
                        output_x = model.Model.classifier(fusion_feature_x)

                        loss = loss_function(output_x, labels)
                        eval_loss += loss.item()
                        y_pred.append(output_x.cpu())
                        y_true.append(labels.cpu())
                        
            eval_loss = eval_loss / len(dataloader)
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = MetricsTop().getMetics(args.datasetName)(pred, true)
            eval_results["Loss"] = round(eval_loss, 4)
            results[n] = eval_results
            logger.info(f"Robustness Test {n}: ({args.model_name}) >> {dict_to_str(eval_results)}")
        return results

    @staticmethod
    def do_robustness_test(model, dataloaders, args):
        loss_function = nn.L1Loss()
        model.eval()
        results = {}
        eval_loss = 0.0
        for n, dataloader in enumerate(dataloaders):
            y_pred, y_true = [], []
            with torch.no_grad():
                with tqdm(dataloaders[dataloader]) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(args.device)
                        audio = batch_data['audio'].to(args.device)
                        text = batch_data['text'].to(args.device)
                        labels = batch_data['labels']['M'].to(args.device).view(-1, 1)

                        fusion_feature_x = model.Model.fusion(text, audio, vision)
                        output_x = model.Model.classifier(fusion_feature_x)

                        loss = loss_function(output_x, labels)
                        eval_loss += loss.item()
                        y_pred.append(output_x.cpu())
                        y_true.append(labels.cpu())
                        
            eval_loss = eval_loss / len(dataloader)
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = MetricsTop().getMetics(args.datasetName)(pred, true)
            eval_results["Loss"] = round(eval_loss, 4)
            results[n] = eval_results
            logger.info(f"Robustness Test {n}: ({args.model_name}) >> {dict_to_str(eval_results)}")
        return results

    def getTrain(self, args):
        return self.TRAIN_MAP[args.modelName.lower()](args)
