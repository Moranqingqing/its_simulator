import argparse
from copy import copy
from pathlib import Path
import json
import pprint
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from itertools import cycle

lines = ["-", "--", "-.", ":"]


def nested_key(d, current_keys):
    for key in d.keys():
        if isinstance(d[key], dict):
            nk = nested_key(d[key], [])
            current_keys.extend(nk)
        current_keys.append(key)
    return current_keys


def reconstruct_grid_search_keys_values(dicts, keys, current_key_path):
    for key in dicts[0]:
        path = copy(current_key_path)
        path.append(key)
        values = set()
        for d in dicts:
            values.add(str(d[key]))
        if len(values) > 1:
            # that means it is grid_search, must go deeper
            if isinstance(d[key], dict):  # nested
                reconstruct_grid_search_keys_values([d[key] for d in dicts], keys, path)
            else:
                keys[key] = (path, values)
        else:
            pass
    return keys


def single_experiment(path):
    p = Path(path)
    data = {}
    for f in p.iterdir():
        if f.is_dir():
            dic_path = str(f.name)
            config_file = f / "params.json"
            result_file = f / "result.json"
            with config_file.open() as f:
                config_dict = json.load(f)
            with result_file.open() as f:
                s = f.readlines()
                s = "[" + ",".join(s) + "]"
                training_iterations = json.loads(s)
            data[dic_path] = (config_dict, training_iterations)

    all_configs_dict = []

    sets = []
    for d in data.values():
        d_keys = nested_key(d[0], [])
        s = set(d_keys)
        sets.append(s)
        all_configs_dict.append(d[0])

    diffset = set().difference(*sets)

    if len(all_configs_dict) > 1 and len(diffset) > 0:
        raise Exception("Keys should match:\n\ndicts:\n\n{}\n\ndiffset:\n\n{}".format(
            "\n\n".format([pprint.pformat(d) for d in all_configs_dict]), "\n".join(diffset)))

    grid_search_keys = reconstruct_grid_search_keys_values(list(all_configs_dict), {}, [])
    # print("grid_search_keys", list(grid_search_keys.keys()))
    return data, grid_search_keys


def get_value(d, path_in_d):
    value = None
    for subkey in path_in_d:
        if value is None:
            if subkey in d:
                value = d[subkey]
        else:
            if subkey in value:
                value = value[subkey]
    return value


def plot(exps, metric_label, metric_path, plot_a_line_n=100):
    fig, ax = plt.subplots(1)
    for exp_label, exp_params in exps.items():
        # print("exp_label", exp_label)
        path = exp_params["path_results"]
        color = exp_params["color"]
        linecycler = cycle(lines)
        if os.path.exists(path):
            dont_analyse = False
            data, grid_search = single_experiment(path)  # todo factorize upper
            if "policy_mapping_fn" in grid_search:
                del (grid_search["policy_mapping_fn"])
            all_runs = {}
            for path, (params, training_iterations) in data.items():

                # print(path, training_iterations)
                label = {}
                for key_param, (path_in_dict, _) in grid_search.items():
                    value = get_value(params, path_in_dict)
                    if value is None:
                        dont_analyse = True
                        break
                    label[key_param] = value
                main_key = str(label)
                # print("main_key", main_key)
                if main_key in all_runs:
                    all_runs[main_key].append(training_iterations)
                else:
                    all_runs[main_key] = [training_iterations]
                # print("all_run.keys()", all_runs.keys())
            if not dont_analyse:
                x = {}
                # stats = {}
                # plot_a_line = False
                for label, runs in all_runs.items():
                    line_style = next(linecycler)
                    # print("line_style", line_style)
                    # print("label: {}".format(label))
                    values_by_run = []

                    for run in runs:
                        values_by_ti = []
                        for ti in run:
                            value = get_value(ti, metric_path)
                            if value is not None:
                                values_by_ti.append(value)

                        values_by_run.append(values_by_ti)

                    x[label] = np.array(values_by_run)

                    # print(label)
                    label_str = exp_label  # "exp={} | ".format(exp[:5])
                    if label:
                        import ast

                        for k, v in ast.literal_eval(label).items():
                            if len(k) > 3:
                                k = k[:3]
                            label_str += " | " + "{}={}".format(k, v)
                    if len(x[label][0]) == 1:
                        # if baseline is just a policy, no need for multiple training iteration
                        mean = np.squeeze(np.mean(x[label], axis=0))

                        var = np.squeeze(np.std(x[label], axis=0))
                        ax.plot(range(plot_a_line_n),
                                [mean] * plot_a_line_n,
                                lw=2,
                                label=label_str,
                                color=color,
                                ls=line_style)
                        ax.fill_between(range(plot_a_line_n),
                                        [mean + var] * plot_a_line_n,
                                        [mean - var] * plot_a_line_n,
                                        alpha=0.2,
                                        color=color)
                    else:
                        if exp_label == "iql_global_reward":
                            # should be removed (and create a better iql_global_reward configuration)
                            x[label] = x[label] / 3.
                        mean = np.mean(x[label], axis=0)
                        var = np.std(x[label], axis=0)
                        ax.plot(range(len(mean)),
                                mean,
                                lw=2,
                                label=label_str,
                                color=color,
                                ls=line_style)
                        ax.fill_between(
                            range(len(mean)),
                            mean + var,
                            mean - var,
                            alpha=0.2,
                            color=color)

        else:
            print("{} does not exists".format(path))

    plt.legend()
    plt.title(metric_label)
    import wolf.utils.os as ooo
    ooo.makedirs("tmp")
    plt.savefig("tmp/" + metric_label)
    plt.show()


EXAMPLE_USAGE = """
example usage:
    python master_plot.py --config example_plot.yaml
"""


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Wolf] Plot multiples single experiments results on a single plot',
        epilog=EXAMPLE_USAGE)

    parser.add_argument('-c', '--config', type=str,
                        default="master_plot.yaml",
                        help='The master plot config file.', required=True)

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    import yaml

    with open(args.config, 'r') as infile:
        d = yaml.full_load(infile)
        pprint.pprint(d)

    for metric_label, metric_params in d["metrics"].items():
        plot(d["experiments"], metric_label, metric_params["path"], d["max_x"])
