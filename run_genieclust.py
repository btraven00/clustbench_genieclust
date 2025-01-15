#!/usr/bin/env python

"""
Omnibenchmark-izes Marek Gagolewski's https://github.com/gagolews/clustering-results-v1/blob/eae7cc00e1f62f93bd1c3dc2ce112fda61e57b58/.devel/do_benchmark_genieclust.py

Takes the true number of clusters into account and outputs a 2D matrix with as many columns as ks tested,
being true number of clusters `k` and tested range `k plusminus 2`
"""

import argparse
import os, sys
import genieclust
import numpy as np

VALID_METHODS = ['genie', 'gic', 'ica']

def load_labels(data_file):
    data = np.loadtxt(data_file, ndmin=1)
    
    if data.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")
    
    return(data)

def load_dataset(data_file):
    data = np.loadtxt(data_file, ndmin=2)
    
    ##data.reset_index(drop=True,inplace=True)
    
    if data.ndim != 2:
        raise ValueError("Invalid data structure, not a 2D matrix?")
    
    return(data)


def do_genie(X, ks):
    res = dict()
    
    genie = genieclust.Genie(
        postprocess="all"
    )
    # how to define the gini thres?
    sys.exit()
    for g in [0.1, 0.3, 0.5, 0.7, 1.0]:
        for K in Ks:
            genie.set_params(gini_threshold=g)
            genie.set_params(n_clusters=K)
            labels_pred = genie.fit_predict(X)+1 # 0-based -> 1-based!!!
            res[K] = labels_pred
    return res

def main():
    parser = argparse.ArgumentParser(description='clustbench geniecluster runner')

    parser.add_argument('--data.matrix', type=str,
                        help='gz-compressed textfile containing the comma-separated data to be clustered.', required = True)
    parser.add_argument('--data.true_labels', type=str,
                        help='gz-compressed textfile with the true labels; used to select a range of ks.', required = True)
    parser.add_argument('--output_dir', type=str,
                        help='output directory to store data files.')
    parser.add_argument('--name', type=str, help='name of the dataset', default='clustbench')
    parser.add_argument('--method', type=str,
                        help='geniecluster method',
                        required = True)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    if args.method not in VALID_METHODS:
        raise ValueError(f"Invalid method `{args.method}`")

    truth = load_labels(getattr(args, 'data.true_labels'))
    k = int(max(truth)) # true number of clusters
    Ks = [k-2, k-1, k, k+1, k+2] # ks tested, including the true number
    
    data = getattr(args, 'data.matrix')
    curr = do_benchmark_fastcluster_range_ks(X= load_dataset(data), Ks = Ks, method = args.method)

    name = args.name

    header=['k=%s'%s for s in Ks]


    curr = np.append(np.array(header).reshape(1,5), curr.astype(str), axis=0)
    np.savetxt(os.path.join(args.output_dir, f"{name}_ks_range.labels.gz"),
               curr, fmt='%s', delimiter=",")#,
               # header = ','.join(header)) 

if __name__ == "__main__":
    main()
