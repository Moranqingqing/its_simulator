from pathlib import Path

from rltl.main.gan import SUPER_GAN
from rltl.utils.experiment_state import ExperimentData

import numpy as np

import matplotlib.pyplot as plt

from rltl.utils.utils_os import makedirs


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def format_errors(errors, params_source, param_test, show_params=False, min_or_max=np.argmax):
    toprint = "" if not show_params else "".join(
        [v + " " if type(v) == str else "{:+.4f} ".format(v) for v in param_test.values()]) + "| "
    # min_idx = np.argmin(errors)
    min_idx = min_or_max(errors)
    # print(errors)
    for isource in range(len(errors)):
        same_env = params_source[isource] == param_test

        if isource == min_idx and same_env:
            toprint += Color.UNDERLINE + Color.BOLD + Color.PURPLE + "{:+5.4f} ".format(
                errors[isource]) + Color.END + Color.END + Color.END
        elif isource == min_idx and not same_env:
            toprint += Color.UNDERLINE + Color.BOLD + "{:+5.4f} ".format(errors[isource]) + Color.END + Color.END
        elif isource != min_idx and same_env:
            toprint += Color.PURPLE + "{:+5.4f} ".format(errors[isource]) + Color.END
        else:
            toprint += "{:+5.4f} ".format(errors[isource])

    diff = ""
    if param_test != params_source[min_idx]:
        for k in param_test.keys():
            va = param_test[k]
            vb = params_source[min_idx][k]
            if va != vb:
                value_env = Color.PURPLE + "{:+.4f}".format(va) + Color.END
                value_better = Color.UNDERLINE + Color.BOLD + "{:+.4f}".format(vb) + Color.END + Color.END
                diff += k + ":" + value_env + "/" + value_better + " "
    return toprint + "\t" + diff


def create_heatmap(title, tab, sources_labels, targets_labels, path, gan_config, show_plot=False):
    makedirs(path)
    tab = np.array(tab)
    # plt.tight_layout()

    fig, ax = plt.subplots()

    im = ax.imshow(tab, aspect='auto')

    # # We want to show all ticks...
    # ax.set_xticks(np.arange(len(params_source)))
    # ax.set_yticks(np.arange(len(params_test)))
    # # ... and label them with the respective list entries
    # ax.set_xticklabels(params_source)
    # ax.set_yticklabels(params_test)
    #
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    #
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(params_test)):
    #     for j in range(len(params_source)):
    #         text = ax.text(j, i, "{:2f}".format(tab[i, j]),
    #                        ha="center", va="center", color="w")

    ax.set_title(title)
    if gan_config["type"] == SUPER_GAN:
        xx = []
        for x in sources_labels:
            if gan_config["factorize_fake_output"]:
                xx.append("{}".format(x))
            else:
                xx.append("real({})".format(x))
                xx.append("fake({})".format(x))
        if gan_config["factorize_fake_output"]:
            x_axis = range(len(sources_labels))
        else:
            x_axis = range(2 * len(sources_labels))
        plt.xticks(x_axis, xx, rotation='vertical')

        y_pos = range(len(targets_labels))
        plt.yticks(y_pos, targets_labels, rotation='horizontal')
    else:
        x_pos = range(len(sources_labels))
        plt.xticks(x_pos, sources_labels, rotation='vertical')
        y_pos = range(len(targets_labels))
        plt.yticks(y_pos, targets_labels, rotation='horizontal')
    # plt.text(1,1,"hello")
    plt.savefig('{}/{}.png'.format(path, title), bbox_inches="tight")  # , dpi=my_dpi * 10)
    if show_plot:
        plt.show()


def array_to_cross_comparaison(tab, params_source, params_test, min_or_max):
    if min_or_max == "min":
        min_or_max = np.argmin
    elif min_or_max == "max":
        min_or_max = np.argmax
    else:
        raise Exception("must be min or max")
    keys = params_source[0].keys()

    toprint = ""
    for ienv in range(len(tab)):
        formaterrors = format_errors(tab[ienv], params_source, params_test[ienv], show_params=True,
                                     min_or_max=min_or_max) + "\n"
        toprint += formaterrors

    len_params = len(
        "".join([v + " " if type(v) == str else "{:+.4f} ".format(v) for v in params_test[0].values()])) + 2

    head = ""  # ""-" * (6+len_params) * len(params_source) + "\n"
    for key in keys:
        xx = " " * len_params
        for param in params_source:
            xx += param[key] + " " if type(param[key]) == str else "{:+5.4f} ".format(param[key])
        head += "{} <- {}\n".format(xx, key)
    head = head + " " * len_params + "-" * 6 * len(params_source) + "\n"

    return head + toprint


def run(C, exp):
    if exp is None:
        exp = ExperimentData(C.path).load()
    for metric in ["relative_acc", "absolute_acc"]:
        title = "Exp={} Sources vs Sources [{}]".format(Path(exp.path).name, metric)
        cc = exp.cc_between_sources[metric]
        print(array_to_cross_comparaison(cc["tab"], cc["source_params"], cc["target_params"], cc["min_or_max"]))
        create_heatmap(title, cc["tab"], cc["source_params"], cc["target_params"], path=exp.path)
        title = "Exp={} Targets vs Sources [{}]".format(Path(exp.path).name, metric)
        print("<<<<<< {} >>>>>>>>".format(title))
        cc = exp.cc_targets[metric]
        print(array_to_cross_comparaison(cc["tab"], cc["source_params"], cc["target_params"], cc["min_or_max"]))
        create_heatmap(title, cc["tab"], cc["source_params"], cc["target_params"], path=exp.path)
