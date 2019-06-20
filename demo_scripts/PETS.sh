#!/bin/bash
python mbexp.py -logdir ./log/PETS \
    -env halfcheetah \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -ca opt-type CEM \
    -ca model-type PE \
    -ca prop-type E
