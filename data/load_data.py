from cgitb import text
import logging
import pickle
from typing import Dict
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ['MMDataset', 'MMDataLoader']

logger = logging.getLogger('MSA')

class testImperfect():

    def __init__(self, args):
        self.seed_list = args.test_seed_list

        self.strategy_map = {
            'random_drop': self.__RANDOM_DROP,
            'frame_drop': self.__FRAME_DROP,
            'block_drop': self.__BLOCK_DROP,
        }
        self.process_func = self.strategy_map[args.test_mode]

    def __RANDOM_DROP(self, text, vision, audio, config: Dict):
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None

        for missing_seed in self.seed_list:
            np.random.seed(missing_seed)
            missing_mask_t = (np.random.uniform(size=input_mask.shape) > config['missing_rate']) * input_mask
            missing_mask_a = (np.random.uniform(size=input_mask.shape) > config['missing_rate']) * input_mask
            missing_mask_v = (np.random.uniform(size=input_mask.shape) > config['missing_rate']) * input_mask


            for i, (mask_t, mask_a, mask_v) in enumerate(zip(missing_mask_t, missing_mask_a, missing_mask_v)):
                mask_t[0] = mask_t[input_len[i]-1] = mask_a[0] = mask_a[input_len[i]-1] = mask_v[0] = mask_v[input_len[i]-1] = 1

            text_m = missing_mask_t * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask - missing_mask_t) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask_a, axis=2) * audio
            vision_m = np.expand_dims(missing_mask_v, axis=2) * vision

            text_m = np.concatenate((np.expand_dims(text_m, 1), text[:,1:,:]), axis=1)

            text_m_multiseed = text_m if text_m_multiseed is None else np.concatenate((text_m_multiseed, text_m), axis=0)
            audio_m_multiseed = audio_m if audio_m_multiseed is None else np.concatenate((audio_m_multiseed, audio_m), axis=0)
            vision_m_multiseed = vision_m if vision_m_multiseed is None else np.concatenate((vision_m_multiseed, vision_m), axis=0)

        return text_m_multiseed, vision_m_multiseed, audio_m_multiseed, input_mask

    def __FRAME_DROP(self, text, vision, audio, config: Dict):
        """ config: missing_rate.
        """
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None

        for missing_seed in self.seed_list:
            np.random.seed(missing_seed)
            missing_mask = (np.random.uniform(size=input_mask.shape) > config['missing_rate']) * input_mask
            
            assert missing_mask.shape == input_mask.shape

            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1
            
            text_m = missing_mask * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask - missing_mask) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask, axis=2) * audio
            vision_m = np.expand_dims(missing_mask, axis=2) * vision

            text_m = np.concatenate((np.expand_dims(text_m, 1), text[:,1:,:]), axis=1) 

            text_m_multiseed = text_m if text_m_multiseed is None else np.concatenate((text_m_multiseed, text_m), axis=0)
            audio_m_multiseed = audio_m if audio_m_multiseed is None else np.concatenate((audio_m_multiseed, audio_m), axis=0)
            vision_m_multiseed = vision_m if vision_m_multiseed is None else np.concatenate((vision_m_multiseed, vision_m), axis=0)

        return text_m_multiseed, vision_m_multiseed, audio_m_multiseed, input_mask

    def __BLOCK_DROP(self, text, vision, audio, config: Dict):
        """ config: missing_rate
        """
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        
        missing_block_len = np.around((input_len - 2) * config['missing_rate']).astype(np.int)

        missing_mask = input_mask.copy()

        for missing_seed in self.seed_list:
            np.random.seed(missing_seed)
            
            # 构造 missing_mask 方法不同，frame_drop 后续操作均一致.
            for i, instance in enumerate(missing_mask):
                start_p = np.random.randint(low=1, high=input_len[i] - missing_block_len[i])
                missing_mask[i, start_p:start_p+missing_block_len[i]] = 0
            
            text_m = missing_mask * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask - missing_mask) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask, axis=2) * audio
            vision_m = np.expand_dims(missing_mask, axis=2) * vision

            text_m = np.concatenate((np.expand_dims(text_m, 1), text[:,1:,:]), axis=1) 

            text_m_multiseed = text_m if text_m_multiseed is None else np.concatenate((text_m_multiseed, text_m), axis=0)
            audio_m_multiseed = audio_m if audio_m_multiseed is None else np.concatenate((audio_m_multiseed, audio_m), axis=0)
            vision_m_multiseed = vision_m if vision_m_multiseed is None else np.concatenate((vision_m_multiseed, vision_m), axis=0)

        return text_m_multiseed, vision_m_multiseed, audio_m_multiseed, input_mask

class MMDataset(Dataset):
    def __init__(self, args, mode='train', **kwarg):
        self.mode = mode
        self.args = args

        self.missing_rate = kwarg['missing_rate'] if mode == 'test' else None
        
        self.augment_rate = {'low': 0.2, 'medium': 0.4, 'high': 0.8}
        self.augment_seed = {'low': 22, 'medium': 222, 'high': 2222}

        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
        }
        DATA_MAP[args.datasetName]()

    def __init_mosi(self):

        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)

        self.text = data[self.mode]['text_bert'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = np.array(data[self.mode]['audio'].astype(np.float32))
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': data[self.mode]['regression_labels'].astype(np.float32)
        }

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        self.audio[self.audio == -np.inf] = 0
        if self.mode == 'train' or self.mode == 'valid':
            if self.args.augment == 'none':
                pass
            elif self.args.augment == 'method_one':
                self.augment_low_text, self.augment_low_vision, self.augment_low_audio, self.text_missing_mask, self.audio_missing_mask, \
                    self.vision_missing_mask, self.text_mask, self.audio_mask, self.vision_mask = \
                    self.augment(self.augment_rate['low'], self.augment_seed['low'], augment_rate=1.0, missing_strategy='structure')
            elif self.args.augment == 'method_two':
                self.augment_low_text, self.augment_low_vision, self.augment_low_audio, self.text_missing_mask, self.audio_missing_mask, \
                    self.vision_missing_mask, self.text_mask, self.audio_mask, self.vision_mask = \
                    self.augment(self.augment_rate['low'], self.augment_seed['low'], augment_rate=1.0, missing_strategy='random')
            elif self.args.augment == 'method_three':
                self.augment_low_text, self.augment_low_vision, self.augment_low_audio, self.text_missing_mask, self.audio_missing_mask, \
                    self.vision_missing_mask, self.text_mask, self.audio_mask, self.vision_mask = \
                    self.augment(self.augment_rate['low'], self.augment_seed['low'], augment_rate=1.0, missing_strategy='block')
            else:
                raise ValueError('UnKnown Augment Strategy!')
            pass
        elif self.mode == 'test':
            """ 测试集对测试的缺失方式采用三随机种子进行缺失构造，为减小缺失位置不同导致的差异。
            """
            processor = testImperfect(self.args)
            self.text, self.vision, self.audio, self.input_mask = processor.process_func(self.text, self.vision, self.audio, {'missing_rate': self.missing_rate})
            self.raw_text = np.concatenate([self.raw_text for i in range(len(self.args.test_seed_list))], axis=0)
            self.ids = list(self.ids) * len(self.args.test_seed_list)
            self.labels['M'] = np.array(list(self.labels['M']) * len(self.args.test_seed_list))

        if  self.args.need_normalized:
            self.__normalize()
    
    def augment(self, missing_rate, missing_seed, augment_rate, missing_strategy='structure'):
        """数据增强方法，构造含有缺失的数据。
        missing_rate: 构造数据的缺失率         missing_seed: 构造数据使用的随机种子      augment_rate: 数据增强方法产生的数据比例
        missing_strategy: 数据增强使用的缺失策略
        """
        input_mask = self.text[:,1,:]
        mask_t = mask_a = mask_v = self.text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
    
        np.random.seed(missing_seed)
        if missing_strategy == 'structure':
            missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask
            missing_mask_t = missing_mask_a = missing_mask_v = missing_mask.copy()
            
            text_m = missing_mask * self.text[:,0,:] + (100 * np.ones_like(self.text[:,0,:])) * (input_mask - missing_mask) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask, axis=2) * self.audio
            vision_m = np.expand_dims(missing_mask, axis=2) * self.vision
            
            # for k, v in enumerate(text_m):
            #     if sum(p) == 0:
            #         text_m[k][0] = 101
            #         self.text[:,1:,:]


            text_m = np.concatenate((np.expand_dims(text_m, 1), self.text[:,1:,:]), axis=1)
        elif missing_strategy == 'random':
            missing_masks = [(np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask for i in range(3)]
            missing_mask_t = missing_masks[0]
            missing_mask_a = missing_masks[1]
            missing_mask_v = missing_masks[2]

            for missing_mask in missing_masks:
                for i, instance in enumerate(missing_mask):
                    instance[0] = instance[input_len[i] - 1] = 1
            
            text_m = missing_masks[0] * self.text[:,0,:] + (100 * np.ones_like(self.text[:,0,:])) * (input_mask - missing_masks[0]) # UNK token: 100.
            audio_m = np.expand_dims(missing_masks[1], axis=2) * self.audio
            vision_m = np.expand_dims(missing_masks[2], axis=2) * self.vision
            
            text_m = np.concatenate((np.expand_dims(text_m, 1), self.text[:,1:,:]), axis=1)
        elif missing_strategy == 'block':
            missing_block_len = np.around((input_len - 2) * missing_rate).astype(np.int)
            missing_mask = input_mask.copy()
            # 构造 missing_mask 方法不同，frame_drop 后续操作均一致.
            for i, instance in enumerate(missing_mask):
                start_p = np.random.randint(low=1, high=input_len[i] - missing_block_len[i])
                missing_mask[i, start_p:start_p+missing_block_len[i]] = 0

            missing_mask_t = missing_mask_a = missing_mask_v = missing_mask
            text_m = missing_mask * self.text[:,0,:] + (100 * np.ones_like(self.text[:,0,:])) * (input_mask - missing_mask) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask, axis=2) * self.audio
            vision_m = np.expand_dims(missing_mask, axis=2) * self.vision
            
            text_m = np.concatenate((np.expand_dims(text_m, 1), self.text[:,1:,:]), axis=1)
        else:
            print("No matched missing strategy.")
            text_m = self.audio
            vision_m = self.vision
            text_m = self.text

        return text_m, vision_m, audio_m, missing_mask_t, missing_mask_a, missing_mask_v, mask_t, mask_a, mask_v

    def __init_mosei(self):
        return self.__init_mosi()

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        if (self.args.augment in ['method_one', 'method_two', 'method_three']) and self.mode != 'test':
            self.augment_low_vision = np.transpose(self.augment_low_vision, (1, 0, 2))
            self.augment_low_audio = np.transpose(self.augment_low_audio, (1, 0, 2))
            # for visual and audio modality, we average across time
            # here the original data has shape (max_len, num_examples, feature_dim)
            # after averaging they become (1, num_examples, feature_dim)
            self.augment_low_vision = np.mean(self.augment_low_vision, axis=0, keepdims=True)
            self.augment_low_audio = np.mean(self.augment_low_audio, axis=0, keepdims=True)

            # remove possible NaN values
            self.augment_low_vision[self.augment_low_vision != self.augment_low_vision] = 0
            self.augment_low_audio[self.augment_low_audio != self.augment_low_audio] = 0

            self.augment_low_vision = np.transpose(self.augment_low_vision, (1, 0, 2))
            self.augment_low_audio = np.transpose(self.augment_low_audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'valid':
            if self.args.augment == 'none':
                sample = {
                    'text': torch.Tensor(self.text[index]), 
                    'audio': torch.Tensor(self.audio[index]),
                    'vision': torch.Tensor(self.vision[index]),
                    'index': index,
                    'id': self.ids[index],
                    'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
                }
            elif self.args.augment in ['method_one', 'method_two', 'method_three']:
                sample = {
                    'text': torch.Tensor(self.text[index]),
                    'text_lm': torch.Tensor(self.augment_low_text[index]),
                    'text_missing_mask': torch.Tensor(self.text_missing_mask[index]),
                    'audio': torch.Tensor(self.audio[index]),
                    'audio_lm': torch.Tensor(self.augment_low_audio[index]),
                    'audio_mask': torch.Tensor(self.audio_mask[index]),
                    'audio_missing_mask': torch.Tensor(self.audio_missing_mask[index]),
                    'vision': torch.Tensor(self.vision[index]),
                    'vision_lm': torch.Tensor(self.augment_low_vision[index]),
                    'vision_mask': torch.Tensor(self.vision_mask[index]),
                    'vision_missing_mask': torch.Tensor(self.vision_missing_mask[index]),
                    'index': index,
                    'id': self.ids[index],
                    'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
                }
            else:
                raise ValueError('UnKnown Augment Strategy!')
        elif self.mode == 'test':
            sample = {
                'text': torch.Tensor(self.text[index]), 
                'audio': torch.Tensor(self.audio[index]),
                'vision': torch.Tensor(self.vision[index]),
                'index': index,
                'id': self.ids[index],
                'raw_text': self.raw_text[index],
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
            }

        return sample

def MMDataLoader(args):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': [MMDataset(args, mode='test', missing_rate=m) for m in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len()

    dataLoader = dict()
    dataLoader['train'] = DataLoader(datasets['train'], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    dataLoader['valid'] = DataLoader(datasets['valid'], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    dataLoader['test'] = [DataLoader(datasets['test'][i], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
                            for i in range(len(datasets['test']))]
    
    return dataLoader