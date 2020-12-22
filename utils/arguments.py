import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description = 'Knowledge Distillation OOD-detection')
    parser_temp = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--source', default = None, type=str, choices = ['W','B','Y','M','F','WM','WF','BM','BF','YM','YF'])
    parser.add_argument('--resume', default = 0, type = int)
    parser.add_argument('--train_mode', default = 'DA', type = str, choices=['DA','SO','TO','CC'])
    # parser.add_argument('--target', default = '', type=str, choices=['W','B','Y','M','F','WM','WF','BM','BF','YM','YF'])
    args = parser.parse_args()
    return args