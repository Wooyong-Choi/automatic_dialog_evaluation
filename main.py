import sys
import argparse
from trainer import *

parser = argparse.ArgumentParser('[*] Argument ')

parser.add_argument('-train',  help='train')

parser.add_argument('-device', help = 'GPU number')
parser.add_argument('-dim',    help = 'embedding dimenstion size')
parser.add_argument('-layer',  help = '# of layer')
parser.add_argument('-batch',  help = '# of batch')
parser.add_argument('-hidden', help = '# of hidden')
parser.add_argument('-margin', help = 'margin')
parser.add_argument('-epoch',  help = '# of epoch')
parser.add_argument('-lr',     help = 'learning rate')

parser.add_argument('-data',   help = 'data folder path')
parser.add_argument(''

parser.add_argument('-test', help = 'test')

args = parser.parse_args()

print(args.train)
