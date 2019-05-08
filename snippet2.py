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
    rd = mcn.data.RandomDataset(
        10000,
        sound_font='mimicopynet/soundfonts/TimGM6mb.sf2',
        inst=[0,1],
        gpu=gpu,
        score_mode='onset',
        lmb_start_const=30.0
    )
    model.load_dataset(rd, None)
    model.load_cqt_inout(None, '1733_raw.npz')
    print('Start learning...')
    model.learn(iter_num=10000000)
    print('Learning Done.')
else: # 推論（耳コピ）モード
    model.load_model("result180505/model_130000.npz")
    print("transcripting from", args.transcript[0], "to", args.transcript[1], "...", end='', flush=True)
    model.transcript(args.transcript[0], args.transcript[1], mode='raw', imgfile=args.transcript[2])
    print("Done.")
