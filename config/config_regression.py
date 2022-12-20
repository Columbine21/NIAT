import os
from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            # single-task
            'niat': self.__NIAT,
            'niat_wo_da': self.__NIAT,
            'niat_wo_dis': self.__NIAT,
            'niat_wo_rec': self.__NIAT,
            'niat_wo_dis_rec': self.__NIAT,
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams(args)

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
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                                 ))

    def __datasetCommonParams(self, args):
        root_dataset_dir = '/home/sharing/lyh/meta_mmsa_yzq/MMSA/data'
        tmp = {
            'mosi': {
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'mosi_aligned.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },
            'mosei': {
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'mosei_aligned.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },
        }
        return tmp

    def __NIAT(self):
        tmp = {
            'commonParas': {
                'need_data_aligned': True, # 使用对齐数据
                'early_stop': 8,
                'need_normalized': False,
                # use finetune for bert
                'use_bert': True,
                'use_bert_finetune': True,
                # module structure selection.
                'fusion': 'structure_one',
                'reconstruction': 'structure_one',
                'discriminator': 'structure_one',
                'classifier': 'structure_one',
            },
            # dataset
            'datasetParas': {
                'mosi': {
                    # temporal convolution kernel size
                    'fus_d_l': 96,
                    'fus_d_a': 24,
                    'fus_d_v': 40,
                    'fus_conv1d_kernel_l': 3,
                    'fus_conv1d_kernel_a': 3,
                    'fus_conv1d_kernel_v': 9,
                    'fus_nheads': 8,
                    'fus_layers': 3,
                    'fus_attn_mask': True,
                    'fus_position_embedding': False,
                    'fus_relu_dropout': 0.0,
                    'fus_embed_dropout': 0.5,
                    'fus_res_dropout': 0.4,
                    'fus_attn_dropout': 0.5,
                    'rec_hidden_dim1': 80,
                    'rec_dropout': 0.4,
                    'rec_hidden_dim2': 96,
                    'disc_hidden_dim1': 128,
                    'disc_hidden_dim2': 64,
                    'clf_dropout': 0.3,
                    'clf_hidden_dim': 80,
                    # train hyperparameter.
                    'alpha': 0.6,
                    'batch_size': 32,
                    'beta': 1.0,
                    'learning_rate': 0.002,
                    'decay': 1e-05,
                    'learning_rate_bert': 2e-05,
                    'learning_rate_other': 0.0005,
                    'weight_decay_bert': 0.0001,
                    'weight_decay_other': 0.0005,
                    'grad_clip': 0.6,
                },
                'mosei': {
                    # temporal convolution kernel size
                    'fus_d_l': 96,
                    'fus_d_a': 16,
                    'fus_d_v': 32,
                    'fus_conv1d_kernel_l': 3,
                    'fus_conv1d_kernel_a': 5,
                    'fus_conv1d_kernel_v': 3,
                    'fus_nheads': 4,
                    'fus_layers': 3,
                    'fus_attn_mask': True,
                    'fus_position_embedding': False,
                    'fus_relu_dropout': 0.5,
                    'fus_embed_dropout': 0.0,
                    'fus_res_dropout': 0.5,
                    'fus_attn_dropout': 0.1,
                    'rec_hidden_dim1': 128,
                    'rec_dropout': 0.2,
                    'rec_hidden_dim2': 64,
                    'disc_hidden_dim1': 80,
                    'disc_hidden_dim2': 32,
                    'clf_dropout': 0.2,
                    'clf_hidden_dim': 256,
                    'alpha': 0.6,
                    'batch_size': 32,
                    'beta': 1.0,
                    'learning_rate': 0.002,
                    'decay': 1e-05,
                    'learning_rate_bert': 2e-06,
                    'learning_rate_other': 0.002,
                    'weight_decay_bert': 0.0,
                    'weight_decay_other': 0.0005,
                    'grad_clip': 1.0,
                },

            },
        }
        return tmp

    def get_config(self):
        return self.args
