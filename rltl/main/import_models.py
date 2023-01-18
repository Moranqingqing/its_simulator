import os

from rltl.utils.utils_os import makedirs, list_checker


def run(C, gan_baselines, classifier_baselines):
    address = C["import_models"]["address"]
    remote_workspace = C["import_models"]["remote_workspace"]
    local_workspace = C["import_models"]["local_workspace"]

    for (key, todo) in [ ("learn_classifiers", classifier_baselines),("learn_gans", gan_baselines)]:
        if key in C["import_models"]["keys"]:
            print("[<<<< {} >>>>>]".format(key))
            baselines = C[key]["baselines"]

            for id_baseline, config_baseline in baselines.items():
                if list_checker(todo, id_baseline):
                    local_models = "{}/{}/{}/".format(local_workspace, key, id_baseline)
                    remote_models = "{}/{}/{}/models".format(remote_workspace, key, id_baseline)
                    makedirs(local_models)
                    cmd = "scp -r {}:{} {}".format(address, remote_models, local_models)
                    print(cmd)
                    os.system(cmd)
