import sys, os, re, math, itertools, copy, csv

from os import listdir
from os.path import isfile, join
from operator import add
from random import shuffle, randint

import numpy as np
from scipy import sparse

from pyspark import SparkContext, SparkConf

TAU = 100

# Range for initializing W and H
MIN_INIT = 0
MAX_INIT = 1

def main():
    if len(sys.argv) < 9:
        print "Not enough arguments."
        sys.exit(0)

    # Parse input parameters
    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])

    beta_value = float(sys.argv[4])
    lambda_value = float(sys.argv[5])

    inputV_filepath = sys.argv[6]
    outputW_filepath = sys.argv[7]
    outputH_filepath = sys.argv[8]

    conf = SparkConf().setAppName("10605hw7").setMaster("local[%d]" % num_workers)
    #conf = SparkConf().setAppName("10605hw7").setMaster("local")
    sc = SparkContext(conf=conf)

    # Behaive differently if input is directory or file
    if os.path.isdir(inputV_filepath):
        file_map = sc.wholeTextFiles(inputV_filepath)
        V_list = file_map.flatMap(map_dir)
    else:
        with open(inputV_filepath) as ff:
            content = ff.readlines()
        file_map = sc.parallelize(content).map(lambda x: x.rstrip('\n'))
        V_list = file_map.map(map_file)

    V_tuples = V_list.map(lambda ((uid, mid), rating): (uid, mid, rating))

    N_i = V_tuples.countByKey()
    N_j = V_tuples.map(lambda (x,y,z): (y,z)).countByKey()

    V_list.cache()

    w_h_key = sc.parallelize(range(num_workers))
    tuple_key = w_h_key.flatMap(lambda x: [(x,y) for y in range(num_workers)])

    # Find maximum user id and movie id
    MAX_UID = V_list.map(lambda ((uid, mid), _): uid).max()
    MAX_MID = V_list.map(lambda ((uid, mid), _): mid).max()

    # Initialize W and H for each partitions
    W_zip = sc.parallelize([(x, np.random.uniform(MIN_INIT, MAX_INIT,
        size=(((MAX_UID - x - 1) / num_workers) + 1, num_factors))) \
                for x in xrange(num_workers)])
    H_zip = sc.parallelize([(x, np.random.uniform(MIN_INIT, MAX_INIT,
        size=(num_factors, ((MAX_MID - x - 1) / num_workers) + 1))) \
                for x in xrange(num_workers)])

    blk_w_size = int(math.ceil(float(MAX_UID) / num_workers))
    blk_w_arr = [(MAX_UID - r - 1) / num_workers + 1 for r in xrange(num_workers)]
    blk_w_arr = [sum(blk_w_arr[:i]) for i in xrange(num_workers)]
    blk_w_rem = MAX_UID % num_workers
    blk_w_cutoff = blk_w_size * blk_w_rem

    blk_h_size = int(math.ceil(float(MAX_MID) / num_workers))
    blk_h_arr = [(MAX_MID - r - 1) / num_workers + 1 for r in xrange(num_workers)]
    blk_h_arr = [sum(blk_h_arr[:i]) for i in xrange(num_workers)]
    blk_h_rem = MAX_MID % num_workers
    blk_h_cutoff = blk_h_size * blk_h_rem

    # Return partition location for rating data
    def get_index(uid, mid):
        if blk_h_cutoff == 0 or mid <= blk_h_cutoff:
            col = (mid - 1) / blk_h_size
        else:
            col = ((mid - blk_h_cutoff - 1) / (blk_h_size - 1)) + blk_h_rem

        if blk_w_cutoff == 0 or uid <= blk_w_cutoff:
            row = (uid - 1) / blk_w_size
        else:
            row = ((uid - blk_w_cutoff - 1) / (blk_w_size - 1)) + blk_w_rem

        return row, col

    # Group rating by partition
    V_zip = V_list.keyBy(lambda ((uid, mid), _): get_index(uid, mid)). \
                map(lambda (x,y): (x,[y])).reduceByKey(add). \
                union(tuple_key.map(lambda x: (x, [])))

    V_zip.cache()

    V_row = V_zip.map(lambda ((row, col), res): ((row, col),
        [uid - blk_w_arr[row] - 1 for ((uid, mid), rating) in list(res)])). \
        reduceByKey(add)
    V_col = V_zip.map(lambda ((row, col), res): ((row, col),
        [mid - blk_h_arr[col] - 1 for ((uid, mid), rating) in list(res)])). \
        reduceByKey(add)
    V_rating = V_zip.map(lambda (key, res): (key,
        [rating for ((uid, mid), rating) in list(res)])). \
        reduceByKey(add)

    # Generate sparse matrix of rating for lookup
    V_mat = V_row.groupWith(V_col, V_rating). \
                map(lambda ((r, c), (row, col, data)):
                    ((r, c), sparse.csr_matrix((list(data)[0], (list(row)[0], list(col)[0])),
                        shape=(((MAX_UID - r - 1) / num_workers) + 1,
                            ((MAX_MID - c - 1) / num_workers) + 1))))

    V_mat.cache()

    iter_count = 0
    j_arr = range(num_workers)

    N_i_bc = sc.broadcast(N_i)
    N_j_bc = sc.broadcast(N_j)
    beta_bc = sc.broadcast(beta_value)
    lambda_bc = sc.broadcast(lambda_value)
    iter_bc = sc.broadcast(iter_count)
    stratum_update = sum(N_i.values()) / num_workers

    def mapStratum(it):
        for x in it:
            w_index = x[0][0]
            h_index = x[0][1]
            V = list(x[1][0])[0]
            W = list(x[1][1])[0]
            H = list(x[1][2])[0]

            yield ((w_index, h_index), dsgd(V, W, H, w_index, h_index,
                beta_bc, lambda_bc, iter_bc, N_i_bc, N_j_bc))

    for i in xrange(num_iterations):
        # For each iteration, shuffle order of strata
        shuffle(j_arr)
        iter_bc = sc.broadcast(iter_count)

        for j in j_arr:
            target_V = V_mat.filter(lambda ((x1, x2), _): x1 == ((x2 + j) % num_workers))
            target_W = W_zip.map(lambda (x, _): ((x, (x - j) % num_workers), _))
            target_H = H_zip.map(lambda (x, _): (((x + j) % num_workers, x), _))

            # Perform SGD for each partition
            #res = target_V.groupWith(target_W, target_H).partitionBy(num_workers). \
            #        mapPartitions(lambda ((w_index, h_index), (V, W, H)): \
            #            ((w_index, h_index), dsgd(list(V)[0], list(W)[0], list(H)[0], w_index, h_index, \
            #                beta_value, lambda_value, iter_count))).collect()
            res = target_V.groupWith(target_W, target_H).partitionBy(num_workers). \
                    mapPartitions(mapStratum)

            W_zip = res.map(lambda ((w_index, h_index), (W_new, H_new, L_loc)): (w_index, W_new))
            H_zip = res.map(lambda ((w_index, h_index), (W_new, H_new, L_loc)): (h_index, H_new))

            # Update W, H, and iteration count
            #W_zip = sc.parallelize([(w_index, W_new) for ((w_index, h_index), (W_new, H_new, iter_count_new, L_loc)) in res])
            #H_zip = sc.parallelize([(h_index, H_new) for ((w_index, h_index), (W_new, H_new, iter_count_new, L_loc)) in res])
            #iter_count += sum([val for (_, (_, _, val, _)) in res])

            iter_count += stratum_update

    W_newzip = W_zip.collect()
    H_newzip = H_zip.collect()

    W_sorted = sorted(W_newzip, key=lambda x: x[0])
    H_sorted = sorted(H_newzip, key=lambda x: x[0])

    # Concatenate W and H to generate final W and H
    W_final = np.concatenate([x[1] for x in W_sorted], axis=0)
    H_final = np.concatenate([x[1] for x in H_sorted], axis=1)

    #print np.dot(W_final, H_final)

    W_csv = open(outputW_filepath, 'w')
    H_csv = open(outputH_filepath, 'w')

    W_write = csv.writer(W_csv, delimiter=',')
    H_write = csv.writer(H_csv, delimiter=',')

    W_write.writerows(W_final)
    H_write.writerows(H_final)

    W_csv.close()
    H_csv.close()

    return

# Function that performs SGD
def dsgd(V, W, H, w_index, h_index, beta_bc, lambda_bc, iter_bc, N_i_bc, N_j_bc):
    beta_value = beta_bc.value
    lambda_value = lambda_bc.value
    iter_count = iter_bc.value
    N_i = N_i_bc.value
    N_j = N_j_bc.value

    L = nzsl(V, W, H, lambda_value)
    L_prev = sys.float_info.max
    V_loc = V.tocoo()
    sgd_count = 0

    # Shuffle data
    data_arr = [(x,y,z) for x,y,z in itertools.izip(V_loc.row, V_loc.col, V_loc.data)]
    shuffle(data_arr)

    for uid, mid, rating in data_arr:
        W_old_row = copy.deepcopy(W[uid, :])
        H_old_col = copy.deepcopy(H[:, mid])

        epsilon_n = math.pow(TAU + iter_count + sgd_count, -beta_value)

        W[uid, :] = W_old_row - epsilon_n * \
            (-2 * (rating - np.dot(W_old_row, H_old_col)) * H_old_col + \
            2 * (lambda_value / V[uid, :].nnz) * np.transpose(W_old_row))
        H[:, mid] = H_old_col - epsilon_n * \
            (-2 * (rating - np.dot(W_old_row, H_old_col)) * np.transpose(W_old_row) + \
            2 * (lambda_value / V[:, mid].nnz) * H_old_col)

        L_prev = L
        L = nzsl(V, W, H, lambda_value)

        # If loss value become larger, undo last action and return
        if L >= L_prev:
            W[uid, :] = W_old_row
            H[:, mid] = H_old_col
            return W, H, sgd_count, L_prev

        sgd_count += 1

    # Return final output
    return W, H, L

# Function for computing loss value
def nzsl(V, W, H, lambda_value):
    res = 0.0
    V_loc = V.tocoo()

    # Compute L_NZSL
    for uid, mid, rating in itertools.izip(V_loc.row, V_loc.col, V_loc.data):
        res += math.pow(rating - np.dot(W[uid, :], H[:, mid]), 2)

    # Add L_2
    res += lambda_value * (sum(np.add.reduce(W * W)) + sum(np.add.reduce(H * H)))

    return res

def map_dir(t):
    arr = t[1].split("\n", 1)
    mid = int(re.findall('\d+', arr[0])[0])
    tmp = [x.split(",") for x in arr[1].split("\n")]
    return [((int(elem[0]), mid), int(elem[1])) for elem in tmp if len(elem) == 3]

def map_file(t):
    arr = t.split(",")
    return ((int(arr[0]), int(arr[1])), int(arr[2]))

if __name__ == "__main__":
    main()
