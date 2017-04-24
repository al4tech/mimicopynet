#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 18:25:35 2016

@author: marshi
"""

import mimicopynet as mcn


#mcn.data.solo_piano_to_wsdata("musicnet.npz", "musicnet_metadata.csv", "wsdata")
#mcn.data.make_cqt_inout("wsdata","testdata.npz")

model = mcn.model.CNN()

#model.load_cqt_inout("testdata.npz")
#model.learn()

model.load_model("result/model_39000.npz")
#model.transcript("Niyodo - piano.wav", "test.mid")
model.transcript("Trivial Lie.wav", "test2.mid")
