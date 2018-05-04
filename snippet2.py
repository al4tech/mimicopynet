import mimicopynet as mcn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--gpu')
parser.add_argument('--transcript', nargs=3)
args = parser.parse_args()


print('loading the model...', end='', flush=True)
gpu = int(args.gpu) if args.gpu is not None else None
model = mcn.model.BasicCNN(input_cnl=2, gpu=gpu)
print('Done.')

if args.transcript is None: # 学習モード
    rd = mcn.data.RandomDataset(300)
    model.load_dataset(rd, None)
    print('Start learning...')
    model.learn(iter_num=10000000)
    print('Learning Done.')
else: # 推論（耳コピ）モード
    model.load_model("result171115/model_1600000.npz")
    print("transcripting from", args.transcript[0], "to", args.transcript[1], "...", end='', flush=True)
    model.transcript(args.transcript[0], args.transcript[1], mode='raw', imgfile=args.transcript[2])
    print("Done.")
