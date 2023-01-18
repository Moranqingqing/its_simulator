from rltl.main import transfer, cc, create_datasets, learn_generative_model, print_transfer_results, \
    env_statistics, import_models, learn_classifiers, import_results, inferences_print_results, inferences, \
    visualize_classifier, threshold_context_sigma
from rltl.utils.configuration import Configuration
from rltl.utils.variant_generator import generate_variants
from rltl.utils.utils_os import save_object, override_dictionary, load_object
import pprint
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('--jobs', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('config_file', type=str, default="configs/debug.yaml", help='path of configuration file')
    parser.add_argument("--show_plots", type=str2bool, nargs='?', const=True,
                        help="Call plt.show() when a plot is created (blocking behavior)", required=False)
    parser.add_argument('--gan_baselines', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('--classifier_baselines', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('--gan_single_envs', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('--dont_load_models', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('--transfer_baselines', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('--envs_to_test', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('--envs_to_generate', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument("--override_config_file", nargs='?', type=str, default=None,
                        help='path of overriding configuration file')

    args = parser.parse_args()
    print(args)
    C = Configuration().load(args.config_file, args.override_config_file).load_tensorflow()
    save_object(C.dict, C["general"]["workspace"] + "/" + C["general"]["id"] + "/" + "config.json", indent=4)
    jobs_list = args.jobs
    show_plots = args.show_plots
    dont_load_models = args.dont_load_models

    if jobs_list is None:
        if "jobs_list" in C["args"]:
            jobs_list = C["args"]["jobs_list"]
        else:
            jobs_list = [
                "create_datasets",
                "env_statistics",
                "learn_generative_models",
                "cross_comparaison",
                "inferences",
                "transfer",
                "print_transfer_results"
            ]

    lists = {
        "classifier_baselines": args.classifier_baselines,
        "gan_baselines": args.gan_baselines,
        "transfer_baselines": args.transfer_baselines,
        "envs_to_test": args.envs_to_test,
        "envs_to_generate": args.envs_to_generate,
        "gan_single_envs": args.gan_single_envs
    }

    for key in lists.keys():

        if lists[key] is None:
            if key in C["args"]:
                lists[key] = C["args"][key]
            else:
                lists[key] = ["all"]
        else:
            if "all" in lists[key]:
                lists[key].clear()
                lists[key].append("all")

    if show_plots is None:
        if "show_plots" in C["args"]:
            show_plots = C["args"]["show_plots"]
        else:
            show_plots = False
    if dont_load_models is None:
        if "dont_load_models" in C["args"]:
            dont_load_models = C["args"]["dont_load_models"]
        else:
            dont_load_models = False

    original_config = load_object(args.config_file)
    if args.override_config_file is None:
        config_overrided = original_config
    else:
        override_config = load_object(args.override_config_file)
        config_overrided = override_dictionary(original_config, override_config)
    for id_variant, (key, config) in enumerate(generate_variants(config_overrided)):
        print("=====================================================")
        print("=====================================================")
        print("grid_search:\n{}".format(pprint.pformat(key)))
        print("=====================================================")
        print("=====================================================")
        config["general"]["id"] = config["general"]["id"] + "/" + str(id_variant)

        jsonable_key = {}
        for k, v in key.items():
            jsonable_key[str(k)] = v

        save_object(jsonable_key,
                    config["general"]["workspace"] + "/" + config["general"]["id"] + "/" + "variant_key.json",
                    indent=4)
        C = Configuration().load(config)
        # C.load(config) #, args.override_config_file)
        C.show_plots = show_plots

        if "create_datasets" in jobs_list:
            create_datasets.run(C, lists["envs_to_generate"])

        if "env_statistics" in jobs_list:
            env_statistics.run(C)

        if "import_models" in jobs_list:
            import_models.run(C, lists["gan_baselines"], lists["classifier_baselines"])

        if "import_results" in jobs_list:
            import_results.run(C)

        if "learn_classifiers" in jobs_list:
            learn_classifiers.run(C, lists["classifier_baselines"])

        if "threshold_context_sigma" in jobs_list:
            threshold_context_sigma.run(C, lists["classifier_baselines"])

        if "visualize_classifier" in jobs_list:
            visualize_classifier.run(C, lists["classifier_baselines"], lists["envs_to_test"])

        if "cross_comparaison_classifier" in jobs_list:
            cc.run(C, lists["classifier_baselines"], "learn_classifiers")  # .save()

        if "learn_generative_models" in jobs_list:
            learn_generative_model.run(C, lists["gan_baselines"], lists["gan_single_envs"])

        if "cross_comparaison" in jobs_list:
            cc.run(C, lists["gan_baselines"], "learn_gans")  # .save()

        if "inferences" in jobs_list:
            inferences.run(C, lists["gan_baselines"], lists["envs_to_test"])  # .save()

        if "inferences_print_results" in jobs_list:
            inferences_print_results.run(C, lists["gan_baselines"], lists["envs_to_test"])

        if "transfer" in jobs_list:
            transfer.run(C, lists["transfer_baselines"], dont_load_models)

        if "print_transfer_results" in jobs_list:
            print_transfer_results.run(C, lists["transfer_baselines"])
