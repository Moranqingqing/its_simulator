import multiprocessing
import os
import logging
import json
import pickle
from pathlib import Path

import yaml
import collections

LOGGER = logging.getLogger(__name__)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool
import time

from random import randint


def list_checker(l, x):
    proceed = l is None or (l is not None and x in l) or "all" in l or not l
    if proceed:
        print("Processing with x={}".format(x))
    # else:
    #     print("Can't proceed x={} (l={})".format(x, l))

    return proceed


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def sleepawhile(t):
    print("Sleeping %i seconds..." % t)
    time.sleep(t)
    return t


def work(num_procs):
    print("Creating %i (daemon) workers and jobs in child." % num_procs)
    pool = multiprocessing.Pool(num_procs)

    result = pool.map(sleepawhile,
                      [randint(1, 5) for x in range(num_procs)])

    # The following is not really needed, since the (daemon) workers of the
    # child's pool are killed when the child is terminated, but it's good
    # practice to cleanup after ourselves anyway.
    pool.close()
    pool.join()
    return result


def test():
    print("Creating 5 (non-daemon) workers and jobs in main process.")
    pool = MyPool(5)

    result = pool.map(work, [randint(1, 5) for x in range(5)])

    pool.close()
    pool.join()
    print(result)


if __name__ == '__main__':
    test()


def makedirs(path):
    LOGGER.info("making directories : {} ...".format(path))
    path = str(path)
    if path != "":
        try:
            os.makedirs(path,exist_ok=True)
        except FileExistsError:
            LOGGER.warning("[FileExistsError] Can't create \"{}\", folder exists".format(path))
    else:
        LOGGER.warning("Folder empty".format(path))


def empty_directory(path_directory):
    os.system("rm -rf {}/*".format(str(path_directory)))


def save_object(object, path, indent=0):
    LOGGER.info("saving object at {}".format(path))

    if isinstance(path, str):
        path = Path(path)
    makedirs(path.parents[0])
    if "json" in path.name:
        with open(path, 'w') as f:
            if indent > 0:
                json_str = json.dumps(object, indent=indent)
            else:
                json_str = json.dumps(object)
            f.write(json_str)
    elif "yaml" in path.name:
        with open(path, 'w') as f:
            yaml.dump(object, f, default_flow_style=False)
    elif "pickle" in path.name:
        with open(path, 'wb') as f:
            pickle.dump(object, f)
    else:
        raise Exception("Unknown backend for {}".format(path))


def load_object(path):
    LOGGER.info("loading object at {}".format(path))


    if type(path) == type(""):
        path = Path(path)

    if type(path) == type({}):
        return path
    else:
        if not path.exists():
            return None
        if "json" in path.name:
            with open(path, 'r') as infile:
                obj = json.load(infile)
        elif "yaml" in path.name:
            with open(path, 'r') as infile:
                obj = yaml.full_load(infile)
        elif "pickle" in path.name:
            with open(path, 'rb') as infile:
                obj = pickle.load(infile)
        else:
            raise Exception("Unknown backend for {}".format(path))
        return obj


def override_dictionary(d, u, copy_d=True):
    if copy_d:
        res = {**d}
    else:
        res = d
    for k, v in u.items():
        if k == '*':
            override_dictionary(res, {k: v for k in res.keys()}, copy_d=False)
            continue
        if isinstance(v, collections.Mapping):
            res[k] = override_dictionary(res.get(k, {}), v, copy_d=False)
        else:
            res[k] = v
    return res


def set_seed(seed, backends=["random", "numpy", "pytorch", "tensorflow"]):
    if seed is not None:
        LOGGER.info("Setting seed = {}".format(seed))
        if "random" in backends:
            import random
            random.seed(seed)
        if "random" in backends:
            import numpy as np
            np.random.seed(seed)
        # if "pytorch" in backends:
        # # import torch
        # if seed is not None:  # pytorch doesn't like None seed
        #     torch.manual_seed(seed)
        if "tensorflow" in backends:
            import tensorflow
            LOGGER.info("tensorflow version={}".format(tensorflow.__version__))
            if int(tensorflow.__version__[0]) >= 2:
                tensorflow.random.set_seed(seed)
            else:
                tensorflow.random.set_random_seed(seed)
        # try:
        #     # setting ray seed
        #     exps = self["ray"]["run_experiments"]["experiments"]
        #     for exp, value in exps.items():
        #         value["config"]["seed"] = seed
        # except Exception:
        #     logging.warning("Seed {} hasn't been set for Ray.".format(seed))
