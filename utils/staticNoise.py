from typing import Dict
import numpy as np

class staticNoise():

    def __init__(self, config: Dict):

        self.seed_list = config['seeds_list']
        self.config = config
        self.strategy_map = {
            'static_random_drop': self.__RANDOM_DROP,
            'temporal_feature_missing': self.__FRAME_DROP,
            'static_block_drop': self.__BLOCK_DROP,
            'static_entire_drop': self.__ENTIRE_DROP,
            'static_antonym_noise': self.__ANTONYM_NOISE,
            'static_delSentiWords_noise': self.__DELSENTIWORDS_NOISE,
            'static_asr_noise': self.__ASR_NOISE,
        }
        self.process_func = self.strategy_map[config['noise_type']]

    def __RANDOM_DROP(self, text, vision, audio):
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        text_mask_multiseed, audio_mask_multiseed, vision_mask_multiseed = None, None, None

        for missing_seed in self.seed_list:
            np.random.seed(missing_seed)
            missing_mask_t = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask
            missing_mask_a = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask
            missing_mask_v = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask

            for i, (mask_t, mask_a, mask_v) in enumerate(zip(missing_mask_t, missing_mask_a, missing_mask_v)):
                mask_t[0] = mask_t[input_len[i]-1] = mask_a[0] = mask_a[input_len[i]-1] = mask_v[0] = mask_v[input_len[i]-1] = 1

            text_m = missing_mask_t * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask - missing_mask_t) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask_a, axis=2) * audio
            vision_m = np.expand_dims(missing_mask_v, axis=2) * vision

            text_m = np.concatenate((np.expand_dims(text_m, 1), text[:,1:,:]), axis=1)

            text_m_multiseed = text_m if text_m_multiseed is None else np.concatenate((text_m_multiseed, text_m), axis=0)
            audio_m_multiseed = audio_m if audio_m_multiseed is None else np.concatenate((audio_m_multiseed, audio_m), axis=0)
            vision_m_multiseed = vision_m if vision_m_multiseed is None else np.concatenate((vision_m_multiseed, vision_m), axis=0)

            text_mask_multiseed = missing_mask_t if text_mask_multiseed is None else np.concatenate((text_mask_multiseed, missing_mask_t), axis=0)
            audio_mask_multiseed = missing_mask_a if audio_mask_multiseed is None else np.concatenate((audio_mask_multiseed, missing_mask_a), axis=0)
            vision_mask_multiseed = missing_mask_v if vision_mask_multiseed is None else np.concatenate((vision_mask_multiseed, missing_mask_v), axis=0)

        return text_m_multiseed, vision_m_multiseed, audio_m_multiseed, text_mask_multiseed, vision_mask_multiseed, audio_mask_multiseed
    
    def __FRAME_DROP(self, text, vision, audio):
        """ config: missing_rate.
        """
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        mask_multiseed = None

        for missing_seed in self.seed_list:
            np.random.seed(missing_seed)
            missing_mask = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask
            
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
            
            mask_multiseed = missing_mask if mask_multiseed is None else np.concatenate((mask_multiseed, missing_mask), axis=0)

        return text_m_multiseed, vision_m_multiseed, audio_m_multiseed, mask_multiseed, mask_multiseed, mask_multiseed

    def __BLOCK_DROP(self, text, vision, audio):
        """ config: missing_rate
        """
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        mask_multiseed = None
        
        missing_block_len = np.around((input_len - 2) * self.config['missing_rate']).astype(np.int)
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
            
            mask_multiseed = missing_mask if mask_multiseed is None else np.concatenate((mask_multiseed, missing_mask), axis=0)

        return text_m_multiseed, vision_m_multiseed, audio_m_multiseed, mask_multiseed, mask_multiseed, mask_multiseed

    def __ENTIRE_DROP(self, text, vision, audio):
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        text_mask_multiseed, audio_mask_multiseed, vision_mask_multiseed = None, None, None

        for missing_seed in self.seed_list:
            np.random.seed(missing_seed)
            missing_mask_t = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate_t']) * input_mask
            missing_mask_a = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate_a']) * input_mask
            missing_mask_v = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate_v']) * input_mask

            for i, (mask_t, mask_a, mask_v) in enumerate(zip(missing_mask_t, missing_mask_a, missing_mask_v)):
                mask_t[0] = mask_t[input_len[i]-1] = mask_a[0] = mask_a[input_len[i]-1] = mask_v[0] = mask_v[input_len[i]-1] = 1

            text_m = missing_mask_t * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask - missing_mask_t) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask_a, axis=2) * audio
            vision_m = np.expand_dims(missing_mask_v, axis=2) * vision

            text_m = np.concatenate((np.expand_dims(text_m, 1), text[:,1:,:]), axis=1)

            text_m_multiseed = text_m if text_m_multiseed is None else np.concatenate((text_m_multiseed, text_m), axis=0)
            audio_m_multiseed = audio_m if audio_m_multiseed is None else np.concatenate((audio_m_multiseed, audio_m), axis=0)
            vision_m_multiseed = vision_m if vision_m_multiseed is None else np.concatenate((vision_m_multiseed, vision_m), axis=0)

            text_mask_multiseed = missing_mask_t if text_mask_multiseed is None else np.concatenate((text_mask_multiseed, missing_mask_t), axis=0)
            audio_mask_multiseed = missing_mask_a if audio_mask_multiseed is None else np.concatenate((audio_mask_multiseed, missing_mask_a), axis=0)
            vision_mask_multiseed = missing_mask_v if vision_mask_multiseed is None else np.concatenate((vision_mask_multiseed, missing_mask_v), axis=0)

        return text_m_multiseed, vision_m_multiseed, audio_m_multiseed, text_mask_multiseed, vision_mask_multiseed, audio_mask_multiseed
    
    def __ANTONYM_NOISE(self, text, vision, audio):
        
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        text_mask_multiseed, audio_mask_multiseed, vision_mask_multiseed = None, None, None

        for missing_seed in self.seed_list:
            np.random.seed(missing_seed)
            missing_mask_t = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask
            missing_mask_a = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask
            missing_mask_v = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask

            for i, (mask_t, mask_a, mask_v) in enumerate(zip(missing_mask_t, missing_mask_a, missing_mask_v)):
                mask_t[0] = mask_t[input_len[i]-1] = mask_a[0] = mask_a[input_len[i]-1] = mask_v[0] = mask_v[input_len[i]-1] = 1

            text_m = missing_mask_t * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask - missing_mask_t) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask_a, axis=2) * audio
            vision_m = np.expand_dims(missing_mask_v, axis=2) * vision

            text_m = np.concatenate((np.expand_dims(text_m, 1), text[:,1:,:]), axis=1)

            text_m_multiseed = text_m if text_m_multiseed is None else np.concatenate((text_m_multiseed, text_m), axis=0)
            audio_m_multiseed = audio_m if audio_m_multiseed is None else np.concatenate((audio_m_multiseed, audio_m), axis=0)
            vision_m_multiseed = vision_m if vision_m_multiseed is None else np.concatenate((vision_m_multiseed, vision_m), axis=0)

            text_mask_multiseed = missing_mask_t if text_mask_multiseed is None else np.concatenate((text_mask_multiseed, missing_mask_t), axis=0)
            audio_mask_multiseed = missing_mask_a if audio_mask_multiseed is None else np.concatenate((audio_mask_multiseed, missing_mask_a), axis=0)
            vision_mask_multiseed = missing_mask_v if vision_mask_multiseed is None else np.concatenate((vision_mask_multiseed, missing_mask_v), axis=0)

        return text_m_multiseed, vision_m_multiseed, audio_m_multiseed, text_mask_multiseed, vision_mask_multiseed, audio_mask_multiseed

    def __DELSENTIWORDS_NOISE(self, text, vision, audio):
        
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        text_mask_multiseed, audio_mask_multiseed, vision_mask_multiseed = None, None, None

        for missing_seed in self.seed_list:
            np.random.seed(missing_seed)
            missing_mask_t = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask
            missing_mask_a = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask
            missing_mask_v = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask

            for i, (mask_t, mask_a, mask_v) in enumerate(zip(missing_mask_t, missing_mask_a, missing_mask_v)):
                mask_t[0] = mask_t[input_len[i]-1] = mask_a[0] = mask_a[input_len[i]-1] = mask_v[0] = mask_v[input_len[i]-1] = 1

            text_m = missing_mask_t * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask - missing_mask_t) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask_a, axis=2) * audio
            vision_m = np.expand_dims(missing_mask_v, axis=2) * vision

            text_m = np.concatenate((np.expand_dims(text_m, 1), text[:,1:,:]), axis=1)

            text_m_multiseed = text_m if text_m_multiseed is None else np.concatenate((text_m_multiseed, text_m), axis=0)
            audio_m_multiseed = audio_m if audio_m_multiseed is None else np.concatenate((audio_m_multiseed, audio_m), axis=0)
            vision_m_multiseed = vision_m if vision_m_multiseed is None else np.concatenate((vision_m_multiseed, vision_m), axis=0)

            text_mask_multiseed = missing_mask_t if text_mask_multiseed is None else np.concatenate((text_mask_multiseed, missing_mask_t), axis=0)
            audio_mask_multiseed = missing_mask_a if audio_mask_multiseed is None else np.concatenate((audio_mask_multiseed, missing_mask_a), axis=0)
            vision_mask_multiseed = missing_mask_v if vision_mask_multiseed is None else np.concatenate((vision_mask_multiseed, missing_mask_v), axis=0)

        return text_m_multiseed, vision_m_multiseed, audio_m_multiseed, text_mask_multiseed, vision_mask_multiseed, audio_mask_multiseed

    def __ASR_NOISE(self, text, vision, audio):
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        text_mask_multiseed, audio_mask_multiseed, vision_mask_multiseed = None, None, None

        for missing_seed in self.seed_list:
            np.random.seed(missing_seed)
            missing_mask_t = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask
            missing_mask_a = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask
            missing_mask_v = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask

            for i, (mask_t, mask_a, mask_v) in enumerate(zip(missing_mask_t, missing_mask_a, missing_mask_v)):
                mask_t[0] = mask_t[input_len[i]-1] = mask_a[0] = mask_a[input_len[i]-1] = mask_v[0] = mask_v[input_len[i]-1] = 1

            text_m = missing_mask_t * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask - missing_mask_t) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask_a, axis=2) * audio
            vision_m = np.expand_dims(missing_mask_v, axis=2) * vision

            text_m = np.concatenate((np.expand_dims(text_m, 1), text[:,1:,:]), axis=1)
            text_m_multiseed = text_m if text_m_multiseed is None else np.concatenate((text_m_multiseed, text_m), axis=0)
            audio_m_multiseed = audio_m if audio_m_multiseed is None else np.concatenate((audio_m_multiseed, audio_m), axis=0)
            vision_m_multiseed = vision_m if vision_m_multiseed is None else np.concatenate((vision_m_multiseed, vision_m), axis=0)

            text_mask_multiseed = missing_mask_t if text_mask_multiseed is None else np.concatenate((text_mask_multiseed, missing_mask_t), axis=0)
            audio_mask_multiseed = missing_mask_a if audio_mask_multiseed is None else np.concatenate((audio_mask_multiseed, missing_mask_a), axis=0)
            vision_mask_multiseed = missing_mask_v if vision_mask_multiseed is None else np.concatenate((vision_mask_multiseed, missing_mask_v), axis=0)
        
        pad_st = np.array([[101,100,102], [1,1,1], [0,0,0]])
        pad = np.concatenate([pad_st, np.zeros([3,47])], axis=-1)
        for k,v in enumerate(text_m_multiseed):
            if v[0][0] == -100:
                text_m_multiseed[k] = pad
        text_m_multiseed = np.array(text_m_multiseed)

        return text_m_multiseed, vision_m_multiseed, audio_m_multiseed, text_mask_multiseed, vision_mask_multiseed, audio_mask_multiseed