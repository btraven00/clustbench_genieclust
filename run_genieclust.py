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

## this maps the ks to their true offset to the truth, e.g.:
# >>> generate_k_range(5)
# {'k-2': 3, 'k-1': 4, 'k': 5, 'k+1': 6, 'k+2': 7}
# >>> generate_k_range(1)
# {'k-2': 2, 'k-1': 2, 'k': 2, 'k+1': 2, 'k+2': 3}
# >>> generate_k_range(2)
# {'k-2': 2, 'k-1': 2, 'k': 2, 'k+1': 3, 'k+2': 4}
## k is the true k
def generate_k_range(k):
    Ks = [k-2, k-1, k, k+1, k+2] # ks tested, including the true number
    replace = lambda x: x if x >= 2 else 2 ## but we never run k < 2; those are replaced by a k=2 run (to not skip the calculation)
    Ks = list(map(replace, Ks))
    
    # ids = ['k-2', 'k-1', 'k', 'k+1', 'k+2']
    ids = list(range(0,5))
    assert(len(ids) == len(Ks))
    
    k_ids_dict = dict.fromkeys(ids, 0)
    for i in range(len(ids)):
        key = ids[i]
        
        k_ids_dict[key] = Ks[i]
    return(k_ids_dict)

def do_genie(X, Ks, g):
    res = dict()
    
    genie = genieclust.Genie(
        postprocess="all"
    )

    # for K in Ks:
    for item in Ks.keys():
        K_id = item  ## just an unique identifier
        K = Ks[K_id] ## the tested k perhaps repeated
        
        genie.set_params(gini_threshold=g)
        genie.set_params(n_clusters=K)
        labels_pred = genie.fit_predict(X)+1 # 0-based -> 1-based!!!
        res[K_id] = labels_pred
    
    return np.array([res[key] for key in res.keys()]).T

def do_gic(X, Ks):
    res = dict()

    for K_id in Ks.keys(): res[K_id] = dict()
        
    for item in Ks.keys():
        K_id = item  ## just an unique identifier
        K = Ks[K_id] ## the tested k perhaps repeated

        
        # do not use compute_all_cuts! - see note on add_clusters in the manual
        gic = genieclust.GIc(
            compute_full_tree=False,
            compute_all_cuts=False,
            postprocess="all")
        
        gic.set_params(n_clusters=K)
        labels_pred = gic.fit_predict(X)+1 # 0-based -> 1-based!!!
        if gic.n_clusters_ == K:
            # due to noise points, some K-partitions might be unavailable
            res[K_id] = labels_pred
        else:
            res[K_id] = np.repeat(K, len(labels_pred)) ## if not good, requested K repeated n times
    return np.array([res[key] for key in res.keys()]).T

def do_ica(X, Ks):
    res = dict()

    for K_id in Ks.keys(): res[K_id] = dict()

    for item in Ks.keys():
        K_id = item  ## just an unique identifier
        K = Ks[K_id] ## the tested k perhaps repeated
         
        ica = genieclust.GIc(
            n_clusters=K,
            compute_full_tree=True,
            postprocess="all",
            compute_all_cuts=True,
            gini_thresholds=[] # this is IcA -- start from n singletons
        )

        labels_pred_matrix = ica.fit_predict(X)+1 # 0-based -> 1-based!!!
        res[K_id] = labels_pred_matrix[K]
    
    return np.array([res[key] for key in res.keys()]).T

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
    parser.add_argument('--gini_threshold', type=str,
                        help='g',
                        required = False)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    if args.method not in VALID_METHODS:
        raise ValueError(f"Invalid method `{args.method}`")
    
    truth = load_labels(getattr(args, 'data.true_labels'))
    k = int(max(truth)) # true number of clusters
    Ks = generate_k_range(k)
    
    data = getattr(args, 'data.matrix')

    if args.method == 'genie':        
        curr = do_genie(X= load_dataset(data), Ks = Ks, g = float(args.gini_threshold))
    elif args.method == 'gic':
        curr = do_gic(X= load_dataset(data), Ks = Ks)
    elif args.method == 'ica':
        curr = do_ica(X= load_dataset(data), Ks = Ks)

    name = args.name

    header=['k=%s'%s for s in Ks.values()]
    
    curr = np.append(np.array(header).reshape(1,5), curr.astype(str), axis=0)
    np.savetxt(os.path.join(args.output_dir, f"{name}_ks_range.labels.gz"),
               curr, fmt='%s', delimiter=",")#,
               # header = ','.join(header)) 

if __name__ == "__main__":
    main()
