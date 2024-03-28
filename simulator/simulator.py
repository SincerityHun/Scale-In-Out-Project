import argparse
import collections
import copy
import glob
import json
import math
import multiprocessing
import os

import numpy as np
import pandas

def simulate(args):
    # Workload Download
    workload = pandas.read_csv(args.workload)
    
    # Set Policy for Cluster
    if args.policy == "default":
        policy = DefaultPolicy()
    elif args.policy == "elastic":
        policy = ElasticPolicy()

    # Set Cluster
    simulator = Cluster(workload, policy, args.min_nodes, num_gpus=args.num_gpus,
                        max_nodes=args.max_nodes, interference=args.interference,
                        low_util=args.low_util, high_util=args.high_util)

if __name__ == "__main__":
    # Argument 받기
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", type=str, help="path to workload csv")
    parser.add_argument("--policy", type=str, default="default",
                        choices=["default","elastic"])
    parser.add_argument("--min-nodes", type=int, default=16,
                        help="min number of nodes in the cluster")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="max number of nodes for cluster autoscaling")
    parser.add_argument("--interval", type=int, default=60,
                        help="scheduling interval in seconds")
    parser.add_argument("--interference", type=float, default=0.0,
                        help="job slowdown due to interference")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="number of GPUs per node")
    parser.add_argument("--low-util", type=float,
                        help="low utility threshold")
    parser.add_argument("--high-util", type=float,
                        help="high utility threshold")
    parser.add_argument("--output", type=str,
                        help="path to output logs")
    args = parser.parse_args()


    if os.path.isdir(args.workload): # 워크로드가 디렉토리일 경우 -> 여러가지
        assert args.output is not None and os.path.isdir(args.output) 
        print("Just File for workload")
    else: # 워크로드가 단 하나일 경우,
        simulate(args)