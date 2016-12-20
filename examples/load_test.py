import time
import multiprocessing as mp
import pickle
import os

DATA_ROOT = '/mnt/share/aai_fukuta/'


def load_pickle(i):
    with open(os.path.join(DATA_ROOT, './data.train_{}.pkl'.format(i)), 'rb') as f:
        kifu = pickle.load(f)
    return kifu


def load_parallel():
    s = time.time()
    pool = mp.Pool()

    data = pool.map(load_pickle, list(range(1, 39)))
    kifu = []
    for x in data:
        kifu.extend(x)
    print(time.time() - s)


def load_serial():
    s = time.time()

    data = []
    for i in range(1, 39):
        data.extend(load_pickle(i))

    print(time.time() - s)


def load_all():
    s = time.time()

    with open(os.path.join(DATA_ROOT, './data.train_all.pkl'), 'rb') as f:
        kifu = pickle.load(f)

    print(time.time() - s)


if __name__ == '__main__':
    load_parallel()
    load_all()
    load_serial()
