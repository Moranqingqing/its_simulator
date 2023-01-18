# from rltl.main.gan import CLASSIC, SUPER_GAN, AGGLOMERATED_GAN
# from rltl.main.policy import RandomPolicy
# from rltl.main.print_cc import create_heatmap
# from rltl.utils.experiment_state import ExperimentData
# from rltl.utils.registry import R
#
# from rltl.utils.replay_memory import Memory
# from rltl.utils.utils_os import makedirs
# from rltl.utils.utils_rl import to_one_hot, extract_not_done_samples, rollouts, collect_steps_from_gan_model
# import numpy as np
# import tensorflow as tf
#
# samples = extract_not_done_samples(n, planning_init_steps)
#                 steps, _ = rollouts(env_creator, policy_for_model, n, k, samples,
#                                     multiprocessing=False)  # , as_tensor=True)
# def run(C, gan_baselines):
#     exp = ExperimentData(C.path, C).load()
#
#     common_parameters = {**C["learn_gans"]}
#     baselines = common_parameters["baselines"]
#     del common_parameters["baselines"]
#
#     labels = []
#     results_reward = {}
#     results_dynamics = {}
#
#     for collection in [exp.source_collection, exp.target_collection]:
#         for i_env, (env_name, (env_creator, config)) in enumerate(collection.envs.items()):
#             labels.append(env_name)
#             print("==============================================")
#             print("==============================================")
#             print("{} {}".format(env_name, config))
#             print("==============================================")
#             print("==============================================")
#             test_env = env_creator()
#             # print("env.horizon={}".format(test_env.horizon))
#             m, _ = rollouts(env_creator,
#                             RandomPolicy(action_space=test_env.action_space),
#                             num_episodes=C["cross_comparaison"]["nb_episodes"],
#                             init_samples=None,
#                             onehot_action=C["onehot_action"],
#                             multiprocessing=False)
#
#             source_samples = m.sample(C["cross_comparaison"]["nb_samples"])
#             S, A, R_, S_ = ([x.s for x in source_samples],
#                             [x.a for x in source_samples],
#                             [x.r_ for x in source_samples],
#                             [x.s_ for x in source_samples])
#             steps = tf.data.Dataset.from_tensor_slices((S, A, R_, S_))
#             steps = steps.batch(len(S))
#
#             for (s, a, r_, s_) in steps:
#                 a = tf.cast(a, tf.float32)
#                 s = tf.cast(s, tf.float32)
#                 s_ = tf.cast(s_, tf.float32)
#                 r_ = tf.cast(r_, tf.float32)
#
#             for id_baseline, config_baseline in baselines.items():
#                 if gan_baselines is None or (gan_baselines is not None and id_baseline in gan_baselines):
#                     if id_baseline not in results_reward:
#                         results_reward[id_baseline] = []
#                         results_dynamics[id_baseline] = []
#                     print("[baseline={}]".format(id_baseline))
#                     baseline = exp.learn_gans_data.baselines[id_baseline]
#                     config_baseline = {**config_baseline, **common_parameters}
#
#                     l1_s_, l1_r_ = test_model(env_name, id_baseline, s, a, r_, s_, C, exp, config_baseline, baseline)
#                     results_dynamics[id_baseline].append(np.mean(l1_s_))
#                     results_reward[id_baseline].append(np.mean(l1_r_))
#             pass
#
#     plot_results(C, "dynamics", labels, results_dynamics)
#     plot_results(C, "reward", labels, results_reward)
#
#
# def plot_results(C, title, labels, results_dynamics):
#     path = C.path / "test_models"
#     makedirs(path)
#     import matplotlib
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     rect_dynamics = {}
#
#     x = np.arange(len(labels))  # the label locations
#
#     width = 0.5  # the width of the bars
#
#     fig, ax = plt.subplots()
#
#     offset = width / len(results_dynamics)
#
#     for i, (baseline, result) in enumerate(results_dynamics.items()):
#         rect_dynamics[baseline] = ax.bar(x - width / 2 + offset * i, result, width / len(results_dynamics),
#                                          label=baseline)
#
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     plt.ylabel('l1')
#     plt.title(title)
#     plt.xticks(x, labels, rotation='vertical')
#     plt.legend()
#     fig.tight_layout()
#     plt.savefig(str(path / (title + ".png")))
#     if C.show_plots:
#         plt.show()
#
#
# def test_model(env_name, id_baseline, s, a, r_, s_, C, exp, config_baseline, baseline):
#     if config_baseline["type"] == CLASSIC:
#         G_dynamics, D_dynamics = None, None
#         if env_name in baseline.envs_dynamics:
#             G_dynamics, D_dynamics = baseline.envs_dynamics[env_name]
#         G_reward, D_reward = None, None
#         if env_name in baseline.envs_reward:
#             G_reward, D_reward = baseline.envs_reward[env_name]
#     elif config_baseline["type"] == AGGLOMERATED_GAN:
#         G_dynamics = baseline.agglomerated_G_dynamics
#         D_dynamics = baseline.agglomerated_D_dynamics
#         G_reward = baseline.agglomerated_G_reward
#         D_reward = baseline.agglomerated_D_reward
#     elif config_baseline["type"] == SUPER_GAN:
#         G_dynamics = baseline.super_G_dynamics
#         D_dynamics = baseline.super_D_dynamics
#         G_reward = baseline.super_G_reward
#         D_reward = baseline.super_D_reward
#     reconstruction_errors_reward = []
#     reconstruction_errors_dynamics = []
#
#     todo = []
#     if D_dynamics is not None:
#         todo.append(("dynamics", s_, reconstruction_errors_dynamics, G_dynamics, D_dynamics))
#     if D_reward is not None:
#         r_new = tf.reshape(r_, (-1, 1))
#         todo.append(("reward", r_new, reconstruction_errors_reward, G_reward, D_reward))
#
#     for key_model, x_, reconstruction_errors, G, D in todo:
#         print("<<<<<<<<<<<<< {} >>>>>>>>>>>>".format(key_model))
#         z_size = config_baseline["z_size"]
#         z = tf.cast(np.random.normal(-1.0, 1.0, size=[len(s), z_size]), tf.float32)
#
#         input = [s, a, z]
#         if config_baseline["type"] == SUPER_GAN:
#             acc_dynamic = D([s, a, x_])
#             if config_baseline["factorize_fake_output"]:
#                 acc_dynamic, _ = acc_dynamic
#             acc_dynamic = np.mean(acc_dynamic, axis=0)
#             tilled_acc_dynamic = tf.tile(tf.expand_dims(acc_dynamic, axis=0), (len(s), 1))
#             input.append(tilled_acc_dynamic)
#         fake_x_ = G(input)
#         for _ in range(100):
#             fake_x_ += G(input)
#         fake_x_ /= 101
#         reconstruction_error = np.abs(x_ - fake_x_)
#         for i in range(5):
#             print("s_={}, fake_s_={}".format(x_[i], fake_x_[i]))
#         reconstruction_errors.append(np.mean(reconstruction_error))
#
#     return reconstruction_errors_dynamics, reconstruction_errors_reward
#
