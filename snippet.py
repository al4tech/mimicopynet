import mimicopynet as mcn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu')
parser.add_argument('--transcript', nargs=2)
parser.add_argument('--mnpath')
args = parser.parse_args()

if args.transcript is None:
    if args.mnpath is not None:
        if os.path.exists('thedata_raw.npz'):
            print('"thedata_raw.npz" exists. skipping generating it.')
        else:
            print('"thedata_raw.npz" does not exist. Generating...')
            mcn.data.musicnet_to_wsdata(args.mnpath + "/musicnet.npz", args.mnpath + "/musicnet_metadata.csv", "wsdata", "Solo Piano")
            mcn.data.make_cqt_inout("wsdata","thedata_raw.npz", mode='raw')
    else:
        print('the argument --mnpath is undefined. skipping generating "thedata_raw.npz".')
else:
    print('[transcript mode]')

print('loading the model...', end='')
gpu = int(args.gpu) if args.gpu is not None else None
model = mcn.model.BasicCNN(input_cnl=2, gpu=gpu)
print('Done.')

if args.transcript is None: # 学習モード
    print('loading "thedata_raw.npz"...', end='')
    model.load_cqt_inout("thedata_raw.npz")
    print('Done.')
    print('Start learning...')
    model.learn()
    print('Learning Done.')
else: # 推論（耳コピ）モード
    model.load_model("result/model_1000000.npz")
    print("transcripting from", args.transcript[0], "to", args.transcript[1], "...", end='')
    model.transcript(args.transcript[0], args.transcript[1])
    print("Done.")
