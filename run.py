import os
import time
import random
import logging
import torch
import pynvml
import argparse
import numpy as np
import pandas as pd

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_tune import ConfigTune
from config.config_regression import ConfigRegression
from utils.functions import assign_gpu, count_parameters, calculate_AUILC

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def set_log(args):
    tune = 'tune' if args.is_tune else 'uniform'
    log_file_path = f'results/logs/{args.modelName}-{args.augment}-{args.datasetName}-{tune}.log'
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
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

def run(args):
    
    if args.is_tune:
        args.model_save_path = os.path.join(args.model_save_dir, 'tune',
                                            f'{args.modelName}-{args.datasetName}-{args.augment}-{args.augment_rate}-{args.seed}.pth')
    else:
        args.model_save_path = os.path.join(args.model_save_dir, 'normals',
                                            f'{args.modelName}-{args.datasetName}-{args.augment}-{args.augment_rate}-{args.seed}.pth')
    if not os.path.exists(os.path.dirname(args.model_save_path)):
        os.makedirs(os.path.dirname(args.model_save_path))
    # indicate used gpu
    args.device = assign_gpu(args.gpu_ids)
    device = args.device
    # load data and models
    dataloader = MMDataLoader(args)
    setup_seed(args.seed)
    model = AMIO(args).to(device)
    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    atio = ATIO().getTrain(args)
    # do train
    atio.do_train(model, dataloader)
    # load pretrained model
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)
    # do test
    if args.is_tune:
        results = ATIO.do_test(model, dataloader['test'], args)
    else:
        results = ATIO.do_test(model, dataloader['test'], args)
        
    return results


def run_tune(args, tune_times=50):
    args.res_save_dir = os.path.join(args.res_save_dir, 'tunes')
    init_args = args
    has_debuged = []  # save used paras
    save_file_path = os.path.join(args.res_save_dir,
                                  f'{args.datasetName}-{args.modelName}-{args.augment}-tune.csv')
    if not os.path.exists(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))

    for i in range(tune_times):
        
        args = init_args
        config = ConfigTune(args)
        args = config.get_config()
        # print debugging params
        logger.info("#"*40 + '%s with %s - (%d/%d)' %
                    (args.modelName, args.augment, i+1, tune_times) + '#'*40)
        for k, v in args.items():
            if k in args.d_paras:
                logger.info(k + ':' + str(v))
        logger.info("#"*90)
        logger.info('Start running %s with %s...' % (args.modelName, args.augment))
        # restore existed paras
        if i == 0 and os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
            for i in range(len(df)):
                has_debuged.append([df.loc[i, k] for k in args.d_paras])
        # check paras
        print(args.d_paras)
        cur_paras = [args[v] for v in args.d_paras]

        if cur_paras in has_debuged:
            logger.info('These paras have been used!')
            time.sleep(3)
            continue

        has_debuged.append(cur_paras)
        results = []
        for j, seed in enumerate([1111, 1112, 1113]):
            args.cur_time = j + 1
            # setup_seed(seed)
            args.seed = seed
            results.append(run(args))
        # save results to csv
        logger.info('Start saving results...')
        if os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
        else:
            df = pd.DataFrame(
                columns=[k for k in args.d_paras] + [k for k in results[0].keys()])
        # stat results
        tmp = [args[c] for c in args.d_paras]
        for col in results[0].keys():
            values = [r[col] for r in results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            tmp.append((mean, std))

        df.loc[len(df)] = tmp
        df.to_csv(save_file_path, index=None)
        logger.info('Results are saved to %s...' % (save_file_path))

def run_normal(args):
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds
    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        # load config
        config = ConfigRegression(args)
        args = config.get_config()
        args.seed = seed
        logger.info('Start running %s with %s...' % (args.modelName, args.augment))
        logger.info(args)
        # runnning
        args.cur_time = i+1
        result = run(args)
        result_cur = dict()
        for k in list(result[list(result.keys())[0]].keys()):
            result_cur[k] = calculate_AUILC([result[v][k] for v in list(result.keys())])
        model_results.append(result_cur)

    criterions = list(model_results[0].keys())
    save_path = os.path.join(args.res_save_dir, f'{args.datasetName}-{args.test_mode}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model", "Augment"] + criterions)
    res = [args.modelName, args.augment]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)

    logger.info('Results are added to %s...' % (save_path))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--need_task_scheduling', type=bool, default=False, help='use the task scheduling module.')
    parser.add_argument('--is_tune', default=False, action='store_true', help='tune parameters ?')
    parser.add_argument('--modelName', type=str, default='niat', help='support niat/niat_wo_da/niat_wo_dis_rec/niat_wo_dis/niat_wo_rec')
    parser.add_argument('--datasetName', type=str, default='mosi', help='support mosi/mosei')
    parser.add_argument('--augment', type=str, default='method_one', help='support none/method_one/method_two/method_three')
    parser.add_argument('--augment_rate', type=int, default=0.2, help='0.1, 0.2, 0.4')
    parser.add_argument('--test_mode', type=str, default='frame_drop', help='support frame_drop/block_drop/random_drop')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/saved_models', help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results', help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[],  help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    parser.add_argument('--test_seed_list', type=list, default=[1, 11, 111, 1111, 11111], help='indicates the seed for test period imperfect construction')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.seeds = [1111,1112,1113]         # 3种子
    logger = set_log(args)
    run_tune(args, tune_times=50) if args.is_tune else run_normal(args)

