import sys
import argparse

parser = argparse.ArgumentParser('[*] Argument ')

parser.add_argument('-train', help='help')
args = parser.parse_args()

print(args.train)
