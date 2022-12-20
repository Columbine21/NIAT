import os
import logging
import torch
import argparse
import numpy as np
import pandas as pd
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.robust_load_data import robustnessTestLoader
from config.config_regression import ConfigRegression
from utils.functions import assign_gpu, setup_seed, calculate_AUILC

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--modelName', type=str, default='niat', help='Name of model', choices=['niat'])
    parser.add_argument('-d', '--datasetName', type=str, default='mosi', choices=['mosi', 'mosei'], help='Name of dataset')
    parser.add_argument('--augmentation', type=str, default='method_one', help='support method_one/method_two/method_three')
    parser.add_argument('--augment_rate', type=int, default=0.2, help='0.1, 0.2, 0.4')
    parser.add_argument('-g', '--gpu-ids', action='append', default=[])
    parser.add_argument('--res_save_dir', type=str, default='results/results', help='path to save results.')
    parser.add_argument('--noise_type', type=str, default='temporal_feature_missing', 
                        help='support temporal_feature_missing/static_block_drop/static_random_drop/static_entire_drop/static_antonym_noise/static_asr_noise/static_delSentiWords_noise')
    parser.add_argument('--model_save_path', type=str, default='results/saved_models', help='dirpath to the pretrained save results.')
    parser.add_argument('--noise_seed_list', type=list, default=[1,11,111,1111,11111], help='indicates the seed for test period imperfect construction')
    return parser.parse_args()

def reproduce(args):

    model_save_path = os.path.join(args.model_save_path, 'normals', f'{args.modelName}-{args.datasetName}-{args.augmentation}-{args.augment_rate}-{args.seed}.pth')

    model = AMIO(args).to(args.device)
    
    dataloaders = robustnessTestLoader(args, num_workers=0)
    logger.info(model_save_path)
    assert os.path.exists(model_save_path)

    model.load_state_dict(torch.load(model_save_path))
    model.to(args.device)

    results = ATIO().do_robustness_test(model, dataloaders, args)

    return results

def run_normal(args):
    config = ConfigRegression(args)
    configs = config.get_config()

    configs.res_save_dir = os.path.join(args.res_save_dir, 'reproduce')

    
    configs['device'] = assign_gpu(args.gpu_ids)
    torch.cuda.set_device(configs['device'])
    configs['res_save_dir'] = os.path.join(args.res_save_dir, 'reproduce')
    configs['augmentation'] = args.augmentation
    configs['augment_rate'] = args.augment_rate
    configs['noise_type'] = args.noise_type
    configs['train_mode'] = 'regression'
    configs['noise_seed_list'] = args.noise_seed_list
    configs['model_save_path'] = args.model_save_path


    model_results = []
    seeds = args.seeds
    # run results
    for i, seed in enumerate(seeds):
        # load config
        setup_seed(seed)
        configs.seed = seed
        logger.info('Start reproducing %s with %s...' % (args.modelName, args.augmentation))
        logger.info(configs)
        # runnning
        configs.cur_time = i+1
        results = reproduce(configs)
        
        if args.noise_type in ['static_asr_noise','static_antonym_noise','static_delSentiWords_noise']:
            result_cur = dict()
            for k in list(results[list(results.keys())[0]].keys()):
                result_cur[k] = ([results[v][k] for v in list(results.keys())])

        elif args.noise_type == 'static_entire_drop':
            result_cur = {
                'T_D': dict(),
                'A_D': dict(),
                'V_D': dict()
            }
            result_cur['T_D'] = results[0]
            result_cur['A_D'] = results[1]
            result_cur['V_D'] = results[2]

        elif args.noise_type in ['temporal_feature_missing', 'static_block_drop', 'static_random_drop']:
            result_cur = dict()
            for k in list(results[list(results.keys())[0]].keys()):
                result_cur[k] = calculate_AUILC([results[v][k] for v in list(results.keys())])
        
        logger.info(f"Result for seed {seed}: ")
        for k in result_cur.keys():
            logger.info(f"{k}: {result_cur[k]}")
        
        model_results.append(result_cur)
        criterions = list(model_results[0].keys()) if args.noise_type != 'static_entire_drop' else list(model_results[0]['T_D'].keys())

    save_path = os.path.join(args.res_save_dir, f'{args.datasetName}-{args.noise_type}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model", "Augmentation", "Test Seeds"] + criterions)
    
    
    if args.noise_type == 'static_entire_drop':
        def d2csv(res, m_res, criterions):
            for c in criterions:
                values = [r[c] for r in m_res]
                mean = round(np.mean(values)*100, 2)
                std = round(np.std(values)*100, 2)
                res.append((mean, std))
            df.loc[len(df)] = res
            df.to_csv(save_path, index=None)

        T_D, A_D, V_D = [], [], []
        for s, v in enumerate(model_results):
            T_D.append(model_results[s]['T_D'])
            A_D.append(model_results[s]['A_D'])
            V_D.append(model_results[s]['V_D'])
        res = [args.modelName, str(args.augmentation) + ' T_Entire Drop', args.noise_seed_list]
        d2csv(res, T_D, criterions)
        res = [args.modelName, str(args.augmentation) + ' A_Entire Drop', args.noise_seed_list]
        d2csv(res, A_D, criterions)
        res = [args.modelName, str(args.augmentation) + ' V_Entire Drop', args.noise_seed_list]
        d2csv(res, V_D, criterions)
        
    else:
        res = [args.modelName, args.augmentation, args.noise_seed_list]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        df.loc[len(df)] = res
        df.to_csv(save_path, index=None)

    logger.info('Results are added to %s...' % (save_path))

if __name__ == '__main__':
    args = parse_args()
    global logger
    def set_log_reproduce(args):
        log_file_path = f'results/logs/{args.modelName}-{args.augmentation}-{args.datasetName}-{args.augment_rate}-test.log'
        # set logging
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        for ph in logger.handlers:
            logger.removeHandler(ph)
        # add FileHandler to log file
        formatter_file = logging.Formatter(
            '%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter_file)
        logger.addHandler(fh)

        return logger
    logger = set_log_reproduce(args)
    args.seeds = [1111, 1112, 1113]         # 3种子
    
    run_normal(args)