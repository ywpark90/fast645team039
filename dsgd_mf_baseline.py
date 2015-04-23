import sys, os, re, math, itertools, copy, csv

from os import listdir
from os.path import isfile, join
from operator import add
from random import shuffle, randint

import numpy as np
from scipy import sparse

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

    V_list = []
    MAX_UID = -1
    MAX_MID = -1

    with open(inputV_filepath) as ff:
        for line in ff:
            arr = [int(x) for x in line.rstrip('\n').split(',')]
            V_list.append(((arr[0], arr[1]), arr[2]))

            if MAX_UID < arr[0]:
                MAX_UID = arr[0]

            if MAX_MID < arr[1]:
                MAX_MID = arr[1]

    # Initialize W and H for each partitions
    W_zip = np.random.uniform(MIN_INIT, MAX_INIT, size=(MAX_UID, num_factors))
    H_zip = np.random.uniform(MIN_INIT, MAX_INIT, size=(num_factors, MAX_MID))

    V_row = [row - 1 for ((row, col), res) in V_list]
    V_col = [col - 1 for ((row, col), res) in V_list]
    V_rating = [res for ((row, col), res) in V_list]

    V_mat = sparse.csr_matrix((V_rating, (V_row, V_col)), shape=(MAX_UID, MAX_MID))

    iter_count = 0
    iteration = 0
    prev_L = sys.float_info.max

    while True:
        W_zip, H_zip, iter_count, L = sgd(V_mat, W_zip, H_zip, beta_value, lambda_value, iter_count)

        if prev_L > L:
            prev_L = L
        else:
            print iteration
            break

        iteration += 1

    #print np.dot(W_zip, H_zip)

    W_csv = open(outputW_filepath, 'w')
    H_csv = open(outputH_filepath, 'w')

    W_write = csv.writer(W_csv, delimiter=',')
    H_write = csv.writer(H_csv, delimiter=',')

    W_write.writerows(W_zip)
    H_write.writerows(H_zip)

    W_csv.close()
    H_csv.close()

    return

# Function that performs SGD
def sgd(V, W, H, beta_value, lambda_value, iter_count):
    L = nzsl(V, W, H, lambda_value)
    L_prev = sys.float_info.max
    V_loc = V.tocoo()

    # Shuffle data
    data_arr = [(x,y,z) for x,y,z in itertools.izip(V_loc.row, V_loc.col, V_loc.data)]
    shuffle(data_arr)

    for uid, mid, rating in data_arr:
        W_old_row = copy.deepcopy(W[uid, :])
        H_old_col = copy.deepcopy(H[:, mid])

        epsilon_n = math.pow(TAU + iter_count, -beta_value)

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
            return W, H, iter_count, L_prev

        iter_count += 1

    # Return final output
    return W, H, iter_count, L

# Function for computing loss value
def nzsl(V, W, H, lambda_value):
    res = 0.0
    V_loc = V.tocoo()

    # Compute L_NZSL
    for uid, mid, rating in itertools.izip(V_loc.row, V_loc.col, V_loc.data):
        res += math.pow(rating - np.dot(W[uid, :], H[:, mid]), 2)

    # Add L_2
    #res += lambda_value * (sum(np.add.reduce(W * W)) + sum(np.add.reduce(H * H)))

    return res

if __name__ == "__main__":
    main()
