import os

from rltl.utils.utils_os import makedirs


def run(C):
    address = C["import_models"]["address"]
    remote_workspace = C["import_models"]["remote_workspace"]
    local_workspace = C["import_models"]["local_workspace"]

    local_models = "{}/test_models".format(local_workspace)

    remote_models = "{}/test_models/dynamics.png".format(remote_workspace)
    makedirs(local_models)
    cmd = "scp {}:{} {}".format(address, remote_models, local_models)
    print(cmd)
    os.system(cmd)


    # remote_models = "{}/test_models/npy".format(remote_workspace)
    # makedirs(local_models)
    # cmd = "scp -r {}:{} {}".format(address, remote_models, local_models)
    # print(cmd)
    # os.system(cmd)

    remote_models = "{}/test_models/worlds".format(remote_workspace)
    makedirs(local_models)
    cmd = "scp -r {}:{} {}".format(address, remote_models, local_models)
    print(cmd)
    os.system(cmd)