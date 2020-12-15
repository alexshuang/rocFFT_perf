#!/usr/bin/env python3

import os
import argparse
import json
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def listify(s):
    if s is None: return s
    elif isinstance(s, list): return s
    elif isinstance(s, tuple): return list(s)
    else: return [s]


class RocProfParser():
    def __init__(self, fpath, num_iter, num_cold_iter, non_header=False):
        if non_header:
            self.df = pd.read_csv(fpath, low_memory=False, header=None)
        else:
            self.df = pd.read_csv(fpath, low_memory=False)
        total_num_iter = num_iter + num_cold_iter
        self.num_kernel = len(self.df) // total_num_iter
        if num_cold_iter > 0:
            self.df = self.df.iloc[num_cold_iter * self.num_kernel:]
        assert(len(self.df) == self.num_kernel * num_iter)
        self.kernel_names = self.df['KernelName'].values[:self.num_kernel]

    def show(self, n=5):
        print(f'Columns: {self.df.columns}')
        print(self.df.iloc[:n])

    def show_avg(self, cols):
        data = []
        cols = listify(cols)
        for n in cols:
            data.append(np.mean(self.df[n].values.reshape(-1, self.num_kernel), axis=0))
        data = np.stack(data, axis=1)
        assert(data.shape[0] == len(self.kernel_names) and data.shape[1] == len(cols))
        for i, (k, row) in enumerate(zip(self.kernel_names, data)):
            res = [f'{n}: {v}' for n, v in zip(cols, row)]
            print('[{}] {}: \n\t{}'.format(i, k, ', '.join(res)))
        
    def show_last(self, cols):
        data = []
        cols = listify(cols)
        for n in cols:
            data.append(self.df[n].values.reshape(-1, self.num_kernel)[-1])
        data = np.stack(data, axis=1)
        assert(data.shape[0] == len(self.kernel_names) and data.shape[1] == len(cols))
        for i, (k, row) in enumerate(zip(self.kernel_names, data)):
            res = [f'{n}: {v}' for n, v in zip(cols, row)]
            print('[{}] {}: \n\t{}'.format(i, k, ', '.join(res)))


def show_profiling(args):
    num_kernel = 0

    if os.path.exists(args.basic_prof_file):
        print("############ BASIC PROF ###############")
        prof = RocProfParser(args.basic_prof_file, 1, 1)
        cols = ["grd","wgr","lds","vgpr","sgpr"]
        cols.extend(args.basic_pmc.split(' '))
        prof.show_last(cols)
        num_kernel = prof.num_kernel
        print("")

    if os.path.exists(args.insts_prof_file):
        print("############ INSTS PROF ###############")
        prof = RocProfParser(args.insts_prof_file, 1, 1)
        cols = args.insts_pmc.split(' ')
        prof.show_last(cols)
        print("")

    if os.path.exists(args.mem_conflict_prof_file):
        print("############ MEMORY CONFLICT PROF ###############")
        prof = RocProfParser(args.mem_conflict_prof_file, 1, 1)
        cols = args.mem_conflict_pmc.split(' ')
        prof.show_last(cols)
        print("")

    if os.path.exists(args.mem_stalled_prof_file):
        print("############ MEMORY STALL PROF ###############")
        prof = RocProfParser(args.mem_stalled_prof_file, 1, 1)
        cols = args.mem_stalled_pmc.split(' ')
        prof.show_last(cols)
        print("")

    print("############ KERNEL DURATION MS ###############");
    assert(num_kernel > 0)
    df = pd.read_csv(args.log_file, low_memory=False, header=None)
    kernel_names = df[2].values[:num_kernel]
    duration_ms = df[4].values.reshape(-1, num_kernel)
    mean_duration_ms = duration_ms[args.num_cold_iter:].mean(0)
    median_duration_ms = np.median(duration_ms[args.num_cold_iter:], axis=0)
    for n, v1, v2 in zip(kernel_names, mean_duration_ms, median_duration_ms):
        print(f"{n}: mean: {v1:.4f}, median: {v2:.4f}")
    print("")

    print("############ END TO END ###############");
    time_pat = re.compile('Execution gpu time: (.*) ms')
    gflops_pat = re.compile('Execution gflops: (.*)')

    data = open(args.perf_file).read()
    times = re.findall(time_pat, data)[0]
    gflops = re.findall(gflops_pat, data)[0]
    times = [eval(o) for o in times.strip().split(' ')][args.num_cold_iter:]
    gflops = [eval(o) for o in gflops.strip().split(' ')][args.num_cold_iter:]
    print(f"Execution gpu time: mean: {np.mean(times):.4f} ms, median: {np.median(times):.4f} ms")
    print(f"Execution gflops: mean: {np.mean(gflops):.4f}, median: {np.median(gflops):.4f}")
    print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = "rocfft performance testing"
    parser.add_argument("--basic_prof_file", type=str)
    parser.add_argument("--basic_pmc", type=str)
    parser.add_argument("--insts_prof_file", type=str)
    parser.add_argument("--insts_pmc", type=str)
    parser.add_argument("--mem_conflict_prof_file", type=str)
    parser.add_argument("--mem_conflict_pmc", type=str)
    parser.add_argument("--mem_stalled_prof_file", type=str)
    parser.add_argument("--mem_stalled_pmc", type=str)
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--perf_file", type=str)
    parser.add_argument("--num_iter", type=int, help='number of run iterations')
    parser.add_argument("--num_cold_iter", type=int, help='number of run iterations')
    parser.add_argument("--batch_count", type=int, help='batch size')
    args = parser.parse_args()

    show_profiling(args)
