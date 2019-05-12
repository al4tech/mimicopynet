# これは原因不明ながらも import fluidsynth が対話モードでだけ成功するため
# 対症療法的に付け加えられたコードです
import sys
sys.__interactivehook__()

import matplotlib
matplotlib.use('Agg')

import glob
import mimicopynet as mcn
import argparse

parser = argparse.ArgumentParser(description='script for learning')
# parser.add_argument('FILENAME', help='input filename')
parser.add_argument('--gpu', type=int, default=-1,
                    help='gpu id')
parser.add_argument('--epoch', type=int, default=2,
                    help='number of epoch')
parser.add_argument('--midiworld-path', type=str, default='../midiworld',
                    help='path to midiworld')
args = parser.parse_args()

train_midis = mcn.data.MidiDataset(
    midis=glob.glob(args.midiworld_path + '/midifile/[Aa]*/*.mid'),
    size=1000,
    sf2_path='mimicopynet/soundfonts/TimGM6mb.sf2',
    length_sec=1.0, no_drum=True, num_part=1
)
eval_midis = mcn.data.MidiDataset(
    midis=glob.glob(args.midiworld_path + '/midifile/[Bb]*/*.mid'),
    size=1000,
    sf2_path='mimicopynet/soundfonts/TimGM6mb.sf2',
    length_sec=1.0, no_drum=True, num_part=1
)

model = mcn.model.NoCQTModel()
model.fit(train_midis, eval_midis=eval_midis, num_epoch=args.epoch, gpu=args.gpu)
