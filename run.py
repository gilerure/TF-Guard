import os
import torch
import random, argparse
import numpy as np

from exp.exp_main import Exp_Main
from exp.exp_bench import Exp_Bench


if __name__ == '__main__':
    # seeding
    seed = 2025    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='TF-Guard & Transformer family for Time Series Anomaly Detection')

    # basic config
    parser.add_argument('--task', type=str, default='train', help='status', choices=['train', 'test', 'detection'])
    parser.add_argument('--model', type=str, default='TFGuard')
    parser.add_argument('--method', type=str, default='main', help='task type', choices=['main', 'benchmark'])

    # supplementary config for FEDFormer model
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # supplementary config for Autoformer model
    parser.add_argument('--factor', type=int, default=1, help='attn factor')

    # data loader
    parser.add_argument('--root_path', type=str, default='dataset/')
    parser.add_argument('--win_size', type=int, default=256, help='window length for detection')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--seq_len', type=int, default=256, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=256, help='start token length')
    parser.add_argument('--pred_len', type=int, default=256, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden layer')
    parser.add_argument('--kernel_size', type=int, nargs='+', default=(8, 16, 32, 64), help='kernel size in Mixture of Seasonals block')
    parser.add_argument('--patch_size', type=int, default=32, help='patching size in Spectral Flux Modeling block')
    parser.add_argument('--K', type=int, default=8, help='top k spectral flux selected')
    parser.add_argument('--threshold', type=float, default=6, help='threshold tau for anomaly detection')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.002, help='optimizer learning rate')  
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for AdamW optimizer')
    parser.add_argument('--epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--step_size', type=int, default=5, help='period of learning rate decay in epochs')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='multiplicative factor to decay learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

    args = parser.parse_args()
    print('>>>>>>>Args in experiment : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    for k, v in sorted(vars(args).items()):
        print('%s: %s' % (str(k), str(v)))

    Exp = Exp_Bench if args.method == 'benchmark' else Exp_Main
    setting = args.model

    if args.task == 'train':
        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
    elif args.task == 'test':
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=True)
    elif args.task == 'detection':
        exp = Exp(args)  # set experiments
        print('>>>>>>>start online detection : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.online_detection(setting)
    else:
        raise ValueError('Please input valid task name')