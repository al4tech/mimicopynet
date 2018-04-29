import mimicopynet as mcn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--gpu')
parser.add_argument('--transcript', nargs=3)
parser.add_argument('--mnpath')
args = parser.parse_args()

if args.transcript is None:
    if args.mnpath is not None:
        if os.path.exists('thedata_train_raw.npz') and os.path.exists('thedata_test_raw.npz'):
            print('"thedata_train_raw.npz" and "thedata_test_raw.npz" exists. skipping generating it.')
        else:
            print('"thedata_train_raw.npz" or "thedata_test_raw.npz" does not exist. Generating...')
            if os.path.exists('wsdata'):
                print('    "wsdata" exists. skipping generating it.')
            else:
                print('    generating "wsdata"...', end='', flush=True)
                mcn.data.musicnet_to_wsdata(args.mnpath + "/musicnet.npz", args.mnpath + "/musicnet_metadata.csv", "wsdata", "Solo Piano")
                print('Done.')
            # 訓練曲とテスト曲を分けずに.npzに保存
            # mcn.data.make_cqt_inout("wsdata","thedata_raw.npz", mode='raw')
            # 訓練曲とテスト曲を分ける．

            wsd_list = glob.glob("wsdata/*.wsd")
            mcn.data.make_cqt_inout(wsd_list[::2],"thedata_train_raw.npz", mode='raw')
            mcn.data.make_cqt_inout(wsd_list[1::2],"thedata_test_raw.npz", mode='raw')
            print('    Done.')

    else:
        print('the argument --mnpath is undefined. skipping generating "thedata_(train|test)_raw.npz".')
else:
    print('[transcript mode]')

print('loading the model...', end='', flush=True)
gpu = int(args.gpu) if args.gpu is not None else None
model = mcn.model.BasicCNN(input_cnl=2, gpu=gpu)
print('Done.')

if args.transcript is None: # 学習モード
    print('loading "thedata_train_raw.npz"...', end='', flush=True)
    model.load_cqt_inout("thedata_train_raw.npz", "thedata_test_raw.npz")
    print('Done.')
    print('Start learning...')
    model.learn(iter_num=10000000)
    print('Learning Done.')
else: # 推論（耳コピ）モード
    model.load_model("result171115/model_1600000.npz")
    print("transcripting from", args.transcript[0], "to", args.transcript[1], "...", end='', flush=True)
    model.transcript(args.transcript[0], args.transcript[1], mode='raw', imgfile=args.transcript[2])
    print("Done.")
