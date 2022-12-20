import logging
import pickle

import numpy as np
import copy
import torch
from torch.utils.data import DataLoader, Dataset
from utils.staticNoise import staticNoise

__all__ = ['MMDataLoader']

logger = logging.getLogger('MMSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train', missing_config=None):
        """ missing config should contain all needed missing info include missing rate, scenarios, and seeds.
        """
        self.mode = mode
        self.args = args
        self.missing_config = missing_config
        if mode == 'train' or mode == 'valid':
            self.augmentation_config = self.args.augmentation_config
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
        }
        DATASET_MAP[args['datasetName']]()

    def __init_mosi(self):
        if self.args.noise_type == 'static_asr_noise':
            data = pickle.load(open('data/noise_data/mosi_asr_aligned.pkl','rb'))
        else:
            with open(self.args.dataPath, 'rb') as f:
                data = pickle.load(f)

        if self.args.get('use_bert', None):
            if self.args.noise_type == 'static_asr_noise':
                self.text = data[self.mode]['text_bert'].astype(np.float32)
            else:
                self.text = data[self.mode]['text_bert'].astype(np.float32)
            # self.args['feature_dims'][0] = 768
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
            # self.args['feature_dims'][0] = self.text.shape[2]
        if self.args['noise_type'] == 'static_delSentiWords_noise':
            with open('data/noise_data/'+self.args['datasetName']+'/mosi_aligned_sentiWords_delete.pkl', 'rb') as f:
                d_n = pickle.load(f)
                self.text_n =  np.array(d_n['text_bert']).astype(np.float32)
                self.text_n[:,1,:] = self.text[:,1,:]

        elif self.args['noise_type'] == 'static_antonym_noise':
            with open('data/noise_data/'+self.args['datasetName']+'/mosi_aligned_sentiWords_antonymReplacement.pkl', 'rb') as f:
                d_n = pickle.load(f)
                self.text_n =  np.array(d_n['text_bert']).astype(np.float32)
                self.text_n[:,1,:] = self.text[:,1,:]

        else:
            self.text_n = ''
        
        self.audio = data[self.mode]['audio'].astype(np.float32)
        # self.args['feature_dims'][1] = self.audio.shape[2]
        self.vision = data[self.mode]['vision'].astype(np.float32)
        # self.args['feature_dims'][2] = self.vision.shape[2]
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }
        
        # audio and visual length.
        if not self.args['need_data_aligned']:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
        else:
            input_mask = self.text[:,1,:]
            self.audio_lengths = self.vision_lengths = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        
        self.audio[self.audio == -np.inf] = 0

        # Augmentation is used for the scenarios where original data is also required. (Static)
        if self.args.get('augmentation', None) and self.mode != 'test':
            if self.augmentation_config['noise_type'].startswith('static'):
                self.noiser = staticNoise(self.augmentation_config)
                self.text_m, self.vision_m, self.audio_m, self.text_missing_mask, self.vision_missing_mask, self.audio_missing_mask = self.noiser.process_func(
                    self.text,
                    self.vision, 
                    self.audio,
                )
            elif self.augmentation_config['noise_type'].startswith('dynamic'):
                self.text_m, self.vision_m, self.audio_m = None, None, None
                self.text_missing_mask, self.vision_missing_mask, self.audio_missing_mask = None, None, None
                
            else:
                raise NotImplementedError

        # Missing config is used for the scenarios where original data is not needed.
        if self.missing_config is not None:
            self.noiser = staticNoise(self.missing_config)
            
            if self.missing_config['noise_type'] in ['static_antonym_noise', 'static_delSentiWords_noise']:
                self.text = self.text_n
            
            self.text, self.vision, self.audio, _, _, _ = self.noiser.process_func(
                self.text,
                self.vision,
                self.audio
            )
            self.labels['M'] = np.array(list(self.labels['M']) * len(self.args.noise_seed_list))
            self.audio_lengths = np.array(list(self.audio_lengths) * len(self.args.noise_seed_list))
            self.vision_lengths = np.array(list(self.vision_lengths) * len(self.args.noise_seed_list))
            self.ids = np.array(list(self.ids) * len(self.args.noise_seed_list))
            self.index = np.array(list(range(len(self.ids) * len(self.args.noise_seed_list))))

        # if self.args.get('need_normalized'):
        #     self.__normalize()

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")
    
    def __init_mosei(self):
        return self.__init_mosi()


    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'audio_lengths': torch.Tensor([self.audio_lengths[index]]),
            'vision_lengths': torch.Tensor([self.vision_lengths[index]]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 

        if self.args.get('augmentation').startswith('static') and self.mode != 'test':
            sample['text_m'] = torch.Tensor(self.text_m[index])
            sample['text_missing_mask'] = torch.Tensor(self.text_missing_mask[index])
            sample['audio_m'] = torch.Tensor(self.audio_m[index])
            sample['audio_missing_mask'] = torch.Tensor(self.audio_missing_mask[index])
            sample['vision_m'] = torch.Tensor(self.vision_m[index])
            sample['vision_missing_mask'] = torch.Tensor(self.vision_missing_mask[index])

        return sample

class MMDatasetMeta(Dataset):
    def __init__(self, args):
        self.args = args
        self.mode = 'valid'
        self.missing_rate_range = args['meta_config']['missing_rate_range']
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
        }
        DATASET_MAP[args['datasetName']]()

    def __init_mosi(self):
        
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)
        
        self.text = data[self.mode]['text_bert'].astype(np.float32)
        self.args['feature_dims'][0] = 768
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.args['feature_dims'][1] = self.audio.shape[2]
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.args['feature_dims'][2] = self.vision.shape[2]
        # self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        self.labels = { 'M': np.array(data[self.mode]['regression_labels']).astype(np.float32) }
        # audio and visual length.
        if not self.args['need_data_aligned']:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
        else:
            input_mask = self.text[:,1,:]
            self.audio_lengths = self.vision_lengths = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        
        self.audio[self.audio == -np.inf] = 0

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")
    
    def __init_mosei(self):
        return self.__init_mosi()

    def _collate_fn(self, batch):

        text = torch.cat([sample['text'].unsqueeze(0) for sample in batch], dim=0)
        index = torch.cat([torch.Tensor([sample['index']]).unsqueeze(0) for sample in batch], dim=0)
        ids = [sample['id'] for sample in batch]
        audio = torch.cat([sample['audio'].unsqueeze(0) for sample in batch], dim=0)
        audio_lengths = torch.cat([sample['audio_lengths'].unsqueeze(0) for sample in batch], dim=0)
        vision = torch.cat([sample['vision'].unsqueeze(0) for sample in batch], dim=0)
        vision_lengths = torch.cat([sample['vision_lengths'].unsqueeze(0) for sample in batch], dim=0)
        label = torch.cat([sample['labels']['M'].unsqueeze(0) for sample in batch], dim=0)
        
        # generate missing mask
        index_batch, ids_batch, audio_lengths_batch, vision_lengths_batch = None, None, None, None
        token_batch, audio_batch, vision_batch, label_batch = None, None, None, None
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        for m_r in self.missing_rate_range:
            # generate missing mask (Random Modality Missing.)
            missing_mask = torch.LongTensor(np.random.uniform(size=input_mask.shape) > m_r) * input_mask
            for i, mask in enumerate(missing_mask):
                mask[0] = mask[input_len[i]-1] = 1

            text_m = missing_mask * text[:,0,:] + torch.LongTensor(100 * np.ones_like(text[:,0,:])) * (input_mask - missing_mask) # UNK token: 100.
            audio_m = (torch.LongTensor(np.expand_dims(missing_mask, axis=2)) * audio).unsqueeze(1)
            vision_m = (torch.LongTensor(np.expand_dims(missing_mask, axis=2)) * vision).unsqueeze(1)

            text_m = torch.cat((text_m.unsqueeze(1), text[:,1:,:]), dim=1).unsqueeze(1).long()
            token_batch = text_m if token_batch is None else torch.cat((token_batch, text_m), dim=1)
            audio_batch = audio_m if audio_batch is None else torch.cat((audio_batch, audio_m), dim=1)
            vision_batch = vision_m if vision_batch is None else torch.cat((vision_batch, vision_m), dim=1)
            index_batch = index if index_batch is None else torch.cat((index_batch, index), dim=0)
            ids_batch = ids if ids_batch is None else ids_batch.extend(ids)
            audio_lengths_batch = audio_lengths if audio_lengths_batch is None else torch.cat((audio_lengths_batch, audio_lengths), dim=0)
            vision_lengths_batch = vision_lengths if vision_lengths_batch is None else torch.cat((vision_lengths_batch, vision_lengths), dim=0)
            label_batch = label if label_batch is None else torch.cat((label_batch, label), dim=1)
            
        return {
            'id': ids_batch,
            'index': index_batch,
            'text': token_batch, 
            'audio': audio_batch,
            'audio_lengths': audio_lengths_batch,
            'vision': vision_batch,
            'vision_lengths': vision_lengths_batch,
            'labels': label_batch
        }

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'audio_lengths': torch.Tensor([self.audio_lengths[index]]),
            'vision_lengths': torch.Tensor([self.vision_lengths[index]]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }
        return sample

def robustnessTestLoader(args, num_workers):
    match args.noise_type:
        # ASR additive
        case 'static_random_drop':
            datasets = {
                f'Random Drop {r}%': MMDataset(args, 'test', missing_config={'noise_type': 'static_random_drop', 'seeds_list': args.noise_seed_list, 'missing_rate': r})
                for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        case 'temporal_feature_missing':
            datasets = {
                f'Temporal Drop {r}%': MMDataset(args, 'test', missing_config={'noise_type': 'temporal_feature_missing', 'seeds_list': args.noise_seed_list, 'missing_rate': r})
                for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        case 'static_block_drop':
            datasets = {
                f'Structured Temporal Drop {r}%': MMDataset(args, 'test', missing_config={'noise_type': 'static_block_drop', 'seeds_list': args.noise_seed_list, 'missing_rate': r})
                for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        case 'static_entire_drop':
            datasets = {
                'Text Modality Drop': MMDataset(args, 'test', missing_config={
                    'noise_type': 'static_entire_drop', 'seeds_list': args.noise_seed_list, 
                    'missing_rate_t': 1.0, 'missing_rate_a': 0.0, 'missing_rate_v': 0.0}),
                'Audio Modality Drop': MMDataset(args, 'test', missing_config={
                    'noise_type': 'static_entire_drop', 'seeds_list': args.noise_seed_list, 
                    'missing_rate_t': 0.0, 'missing_rate_a': 1.0, 'missing_rate_v': 0.0}),
                'Vision Modality Drop': MMDataset(args, 'test', missing_config={
                    'noise_type': 'static_entire_drop', 'seeds_list': args.noise_seed_list,
                    'missing_rate_t': 0.0, 'missing_rate_a': 0.0, 'missing_rate_v': 1.0}),
            }
        case 'static_antonym_noise':
            datasets = {
                f'Text Modality Antonym Noise': MMDataset(args, 'test', missing_config={'noise_type': 'static_antonym_noise', 'seeds_list': args.noise_seed_list, 'missing_rate': 0.0})
            }
        case 'static_delSentiWords_noise':
            datasets = {
                f'Text Modality DelSentiWords Noise': MMDataset(args, 'test', missing_config={'noise_type': 'static_delSentiWords_noise', 'seeds_list': args.noise_seed_list, 'missing_rate': 0.0})
            }
        case 'static_asr_noise':
            datasets = {
                f'Text Modality ASR Noise': MMDataset(args, 'test', missing_config={'noise_type': 'static_asr_noise', 'seeds_list': args.noise_seed_list, 'missing_rate': 0.0})
            }
        case 'static_random_additive':
            datasets = {
                f'Structured Temporal Drop {r}%': MMDataset(args, 'test', missing_config={'noise_type': 'static_random_additive', 'seeds_list': args.noise_seed_list, 'missing_rate': r})
                for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        case _:
            logger.info(f'Unknown noise type: {args.noise_type}')

    dataloaders = {n: DataLoader(dataset, batch_size=args['batch_size'], shuffle=False, num_workers=num_workers) for n, dataset in datasets.items()}
    return dataloaders

def MMDataLoader(args, num_workers):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader


