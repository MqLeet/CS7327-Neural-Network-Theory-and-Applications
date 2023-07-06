#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/11 16:56     mql        1.0         None
'''
import argparse
from train import train_mlp, train_mlqp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="MLQP", help="Valid model: 'MLP', 'MLQP'")
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="num of hidden layers")
    parser.add_argument("--hidden_dim", type=int, default=32, help="num of neurons")
    parser.add_argument("--epoches", type=int, default=10000)
    parser.add_argument("--name", type=str, default="mlp_lr1e-1", help="experiment name")

    args = parser.parse_args()

    if args.model_name == "MLP":
        train_mlp(num_hidden_layers=args.num_hidden_layers, input_dim=args.input_dim, output_dim=args.output_dim,
                  hidden_dim=args.hidden_dim, name=args.name, epoches=args.epoches, lr=args.learning_rate)

    if args.model_name == "MLQP":
        train_mlqp(num_hidden_layers=args.num_hidden_layers, input_dim=args.input_dim, output_dim=args.output_dim,
                  hidden_dim=args.hidden_dim, name=args.name, epoches=args.epoches, lr=args.learning_rate)

