import os
import random
import time

from utils.functions import Storage


class ConfigTune():
    def __init__(self, args):
        # global parameters for running
        self.globalArgs = args
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            # single-task
            'niat': self.__NIAT,
        }
        # set random seed
        random.seed(int(time.time()))
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()
        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]

        dataArgs = dataArgs['aligned'] if (
            commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['debugParas'],
                                 ))

    def __datasetCommonParams(self):
        root_dataset_dir = '/home/sharing/disk3/Datasets/MMSA-Standard'
        tmp = {
            'mosi': {
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'H': 3.0
                },
            },
            'mosei': {
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'H': 3.0
                }
            }
        }
        return tmp

    def __NIAT(self):
        tmp = {
            'commonParas': {
                'need_data_aligned': True, # 使用对齐数据
                
                'need_normalized': False,
                # use finetune for bert
                'use_bert_finetune': False,
                # module structure selection.
                'fusion': 'structure_one',
                'fus_attn_mask': True,
                'reconstruction': 'structure_one',
                'discriminator': 'structure_one',
                'classifier': 'structure_one',
            },
            'debugParas': {
                'd_paras': ['early_stop', 'fus_d_l', 'fus_d_a', 'fus_d_v', 'fus_conv1d_kernel_l', 'fus_conv1d_kernel_a', 'fus_conv1d_kernel_v',
                            'fus_nheads', 'fus_layers', 'fus_relu_dropout', 'fus_embed_dropout', 'fus_res_dropout', 'fus_attn_dropout', 'fus_position_embedding',
                            'rec_hidden_dim1', 'rec_dropout', 'rec_hidden_dim2', 
                            'disc_hidden_dim1', 'disc_hidden_dim2',
                            'clf_dropout', 'clf_hidden_dim', 
                            'alpha', 'batch_size', 'beta', 'learning_rate', 'decay',
                            'learning_rate_bert', 'learning_rate_other', 'weight_decay_bert', 'weight_decay_other', 'grad_clip'],
                'early_stop': random.choice([6, 8]),
                'fus_d_l':random.choice([96, 120, 256]),
                'fus_d_a':random.choice([16, 32]),
                'fus_d_v':random.choice([16, 32]),
                'fus_conv1d_kernel_l':random.choice([3, 5, 9]),
                'fus_conv1d_kernel_a':random.choice([3, 5, 9]),
                'fus_conv1d_kernel_v':random.choice([3, 5, 9]),
                'fus_nheads':random.choice([8]),
                'fus_layers':random.choice([3,4,5,6]),
                'fus_relu_dropout': random.choice([0.0, 0.1, 0.2, 0.3, 0.4]),
                'fus_embed_dropout': random.choice([0.0, 0.1, 0.2, 0.3, 0.4]),
                'fus_res_dropout': random.choice([0.0, 0.1, 0.2, 0.3, 0.4]),
                'fus_attn_dropout': random.choice([0.0, 0.1, 0.2, 0.3, 0.4]),
                'fus_position_embedding': random.choice([True, False]),
                'rec_hidden_dim1': random.choice([80, 64]),
                'rec_dropout': random.choice([0.4]), 
                'rec_hidden_dim2': random.choice([96, 64]),
                'disc_hidden_dim1': random.choice([128, 64]),
                'disc_hidden_dim2': random.choice([64, 32]),
                'clf_dropout': random.choice([0.3]), 
                'clf_hidden_dim': random.choice([80, 64]),
                'alpha': random.choice([0.2, 0.4, 0.6, 0.8]),
                'batch_size': random.choice([64]),
                'beta': random.choice([0.5, 0.8, 1.0]),
                'learning_rate': random.choice([5e-4, 1e-3, 2e-3, 5e-3]),
                'decay': random.choice([1e-05]),
                'learning_rate_bert': random.choice([1e-05, 5e-6, 2e-5]),
                'learning_rate_other': random.choice([5e-4, 1e-3, 2e-4, 2e-3]),
                'weight_decay_bert': random.choice([0.0, 0.0001, 0.0002, 0.00005]),
                'weight_decay_other': random.choice([0.0, 0.002, 0.001, 0.0005]),
                'grad_clip': random.choice([0.6, 0.8, 1.0]),
            }
        }
        return tmp

    def get_config(self):
        return self.args
