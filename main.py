import sys
import argparse
from trainer import train, test

parser = argparse.ArgumentParser('[*] Argument ')

parser.add_argument('-train', default = 'true', help = 'train True or False')
parser.add_argument('-test',  default = 'false',help = 'test True or False')

parser.add_argument('-device', default = 3,     help = 'GPU number')
parser.add_argument('-dim',    default = 150,   help = 'embedding dimenstion size')
parser.add_argument('-layer',  default = 1,     help = '# of layer')
parser.add_argument('-batch',  default = 1028,  help = '# of batch')
parser.add_argument('-hidden', default = 512,   help = '# of hidden')
parser.add_argument('-margin', default = 1,     help = 'margin')
parser.add_argument('-epoch',  default = 50,    help = '# of epoch')
parser.add_argument('-lr',     default = 0.001, help = 'learning rate')

parser.add_argument('-data',   default = './dataset/',    help = 'data folder path')
parser.add_argument('-pretrain',default= './sample_onmt/',help = 'pretrain model path')
parser.add_argument('-output', default = './result/model',help = 'output path')

args = parser.parse_args()

if args.train == 'true':
    train(args)

if args.test is 'true':
    test(args)

print('[*] OVER')
