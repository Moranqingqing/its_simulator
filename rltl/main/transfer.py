from __future__ import absolute_import, division, print_function
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from rltl.envs.gridworld.envgridworld import EnvGridWorld
from rltl.main.dqn_agent import DQNAgent
from rltl.main.gan import generator, discriminator, perform_an_epoch, \
    create_plots_and_writers, SUPER_GAN, CLASSIC, AGGLOMERATED_GAN
from rltl.main.learn_generative_model import get_gym_size, create_ds
from rltl.main.policy import RandomPolicy, ExponentialDecay, EpsilonGreedyPolicy
from rltl.utils.experiment_state import ExperimentData
from rltl.utils.registry import R
from rltl.utils.replay_memory import Memory
from rltl.utils.transition import TransferTransition
from rltl.utils.utils_os import makedirs, override_dictionary
from rltl.utils.utils_rl import EnvWrapper, rollouts, to_one_hot, extract_not_done_samples, \
    collect_steps_from_gan_model


class Dynamics:

    def __init__(self, transfer_model, target_model, context_model, fn_get_areas):
        """

        Args:
            transfer_model:
            target_model:
            context_model:
            fn_get_areas: return several areas for a given s, should be O(1) (areas overlapping with gridlike cut for example)
        """
        self.transfer_model = transfer_model
        self.get_area = fn_get_areas
        self.target_model = target_model
        self.context_model = context_model

        """
        area0 and area1 overlap
        
        area0: 
            action0
                test_samples: [sample, sample ...]            
                train_samples: [sample, sample ...]
                todo_samples: [sample, sample ...]
                statistics:
                    target_test_samples:
                        mu
                        std                        
                
                    target_model:                    
                        mean'
                        std'
                        skew
                        kurt'
                    transfer_model:
                        mean
                        ...    
            }
        area1:
            ...
            
        """
        self.areas = {}

    def update_target_model_statistics(self):
        # TODO TODO TODO
        # # update stats, do moving average (but must keep new untested samples)
        # all_samples = []
        # for each area:
        #     for action:
        #
        #         todo_samples = [sample, sample ..]
        #         all_samples.append_all(todo_samples)
        #         todo_samples.clear()
        #
        # # use all_samples to do the forwarding for transfer model

        # or just do brute force on all samples (bug-free but no optimisation)

        pass

    def forward(self, s, a):
        area_key = self.get_area(self, s)
        if area_key not in self.areas:
            return None
        else:
            area = self.areas[area_key]
            if tuple(a) not in area:
                return None
            else:
                action_data = area[tuple(a)]
                train_samples = action_data["train_samples"]
                test_samples = action_data["test_samples"]
                if train_samples and test_samples:
                    gt_stats = action_data["statistics"]["target_model"]
                    target_stats = action_data["statistics"]["target_test_samples"]
                    transfer_stats = action_data["statistics"]["transfer_model"]
                    mu_gt = gt_stats["mu"]
                    skew_gt = gt_stats["skew"]
                    std_gt = gt_stats["std"]
                    kurt_gt = gt_stats["kurt"]
                    mu_target = abs(gt_stats[mu_gt - target_stats["mu"]]) / abs(mu_gt)
                    mu_transfer = abs(mu_gt - transfer_stats["mu"]) / abs(mu_gt)
                    std_target = abs(std_gt - target_stats["std"]) / abs(std_gt)
                    std_transfer = abs(std_gt - transfer_stats["std"]) / abs(std_gt)
                    skew_target = abs(skew_gt - target_stats["skew"]) / abs(skew_gt)
                    skew_transfer = abs(skew_gt - transfer_stats["skew"]) / abs(skew_gt)
                    kurt_target = abs(kurt_gt - target_stats["kurt"]) / abs(kurt_gt)
                    kurt_transfer = abs(kurt_gt - transfer_stats["kurt"]) / abs(kurt_gt)
                    target_error = mu_target + std_target + skew_target + kurt_target
                    transfer_error = mu_transfer + std_transfer + skew_transfer + kurt_transfer
                    use_transfer_model = target_error > transfer_error
                elif train_samples and not test_samples:
                    use_transfer_model = True
                elif not train_samples and test_samples:
                    use_transfer_model = True
                else:
                    raise Exception("Impossible")
                if use_transfer_model:
                    rdm_sample = test_samples[np.random.randint(0, len(test_samples))]
                    c = self.context_model(rdm_sample.s, rdm_sample.a, rdm_sample.s_)
                    # TODO must compute std_c to know if sample onehot c and use as it is
                    return self.transfer_model(s, a, c)
                else:
                    return self.target_model(s, a)

    def push_target_sample(self, sample, train_or_test):  # or test
        if not train_or_test in ["train", "test"]:
            raise Exception("train_or_test must be 'train' or 'test'")
        area = self.get_area(self, sample.s)

        # if area in self.areas:
        #
        #     # if tuple(sample.a) in self.areas[area]:
        #
        #     # else:
        #
        # else:
        #     # create it with

        self.areas[area][sample.action][train_or_test + "_samples"].append(sample)
        if train_or_test == "test":
            self.areas[area][sample.action]["todo_samples"].append(sample)


num_cores = 4

DQN = "dqn"
DQN_EXTRA_UPDATES = "dqn_extra_updates"
DQN_EXTRA_STEPS = "dqn_extra_steps"
SOLUTION_1 = "solution_1"
SOLUTION_3 = "solution_3"
SOLUTION_2 = "solution_2"
DYNA_MODEL_LEARNING = "dyna_model_learning"
DYNA_AGGLO = "dyna_agglo"
DYNA_RDM = "dyna_rdm"
DYNA_IMPERFECT = "dyna_pretrained"
DYNA_PERFECT = "dyna_perfect"


def run(C, transfer_baselines, dont_load_models=False):
    exp = ExperimentData(C.path, C).load(dont_load_models=True)

    learn_gans = {**C["learn_gans"]}
    learn_gan_baseline = learn_gans["baselines"]
    del learn_gans["baselines"]
    common_parameters = {**C["transfer"]}
    baselines = common_parameters["baselines"]
    del common_parameters["baselines"]
    # such that we start the result for all the baselines for the same env
    k = 0
    for i_run in range(C["transfer"]["nb_runs"]):
        for ic, collection in enumerate([exp.source_collection, exp.target_collection]):
            for env_name, config in collection.envs.items():
                args_pool = []

                for id_baseline, transfer_config in baselines.items():
                    if transfer_baselines is None or (
                            transfer_baselines is not None and id_baseline in transfer_baselines):
                        argsss = (C, i_run,
                                  id_baseline,
                                  env_name,
                                  {**learn_gan_baseline[transfer_config["gan_baseline"]],
                                   **learn_gans} if "gan_baseline" in transfer_config else None,
                                  override_dictionary(common_parameters, transfer_config),
                                  # {**transfer_config, **common_parameters},
                                  "[RUN={}][collection={}][env_name={} / config={}][baseline={}]"
                                  .format(i_run, ["source", "target"][ic], env_name, config, id_baseline),
                                  dont_load_models)
                        args_pool.append(argsss)
                        k += 1
                if C.multiprocessing:
                    # from multiprocessing import Pool
                    from ray.util.multiprocessing import Pool
                    with Pool(len(args_pool)) as p:
                        p.starmap(run_on_env, args_pool)
                else:
                    for argss in args_pool:
                        run_on_env(*argss)
    if k == 0:
        print("[WARNING] Nothing to do")


def run_on_env(C, i_run, id_baseline, env_name, config_gan_baseline, config_transfer_baseline, str_to_print,
               dont_load_models):
    # print(C)
    print(str_to_print)
    # C = Configuration()
    # print(C)
    # C.load(C_config)
    # print("flag3")
    makedirs(C.path / "transfer" / "worlds")
    path = C.path / "transfer" / id_baseline / env_name / "run_{}".format(i_run)
    tf_log_path = C.path / "transfer" / "tf_log"
    exp = ExperimentData(C.path,
                         C).load(
        dont_load_models=dont_load_models)  # forced to do that here, because tensorflow model are not serializable for Pool
    path = pathlib.Path(path)
    tf_log_path = pathlib.Path(tf_log_path)
    # ugly , quick refractoring for multiprocessing
    makedirs(path)
    if "gan_baseline" in config_transfer_baseline:
        gan_baseline = exp.learn_gans_data.baselines[config_transfer_baseline["gan_baseline"]]
    else:
        gan_baseline = None
    # =============================
    # parameters
    # =============================
    use_stop_transfer_mechanism = "stop_transfer_mechanism" in config_transfer_baseline and config_transfer_baseline[
        "stop_transfer_mechanism"]
    n, k = tuple(config_transfer_baseline["shape"])
    baseline = config_transfer_baseline["baseline"]
    onehot_action = C["onehot_action"]
    run_tensorboard = C["general"]["run_tensorboard"]
    training_interval = config_transfer_baseline["training_interval"]
    update_network_frequency = config_transfer_baseline["update_network_frequency"]
    optimizer = config_transfer_baseline["optimizer"]
    gamma = config_transfer_baseline["gamma"]
    stop_mean_return = config_transfer_baseline["stop_mean_return"]
    use_true_done_to_stop_prediction = config_transfer_baseline["use_true_done_to_stop_prediction"]
    use_true_reward_prediction = config_transfer_baseline["use_true_reward_prediction"]
    action_selection_for_model_prediction = config_transfer_baseline["action_selection_for_model_prediction"]
    update_model_interval = config_transfer_baseline["update_model_interval"]
    update_accuracy_interval = config_transfer_baseline["update_accuracy_interval"]
    plot_interval = config_transfer_baseline["plot_interval"]
    check_stop_transfer_interval = config_transfer_baseline["check_stop_transfer_interval"]
    log_target_model_performance_interval = config_transfer_baseline["log_target_model_performance_interval"]
    mini_batch_size = config_transfer_baseline["minibatch_size"]
    initial_collect_steps = config_transfer_baseline["initial_collect_steps"]
    num_real_steps = config_transfer_baseline["num_real_steps"]
    replay_buffer_max_length = config_transfer_baseline["replay_buffer_max_length"]
    planning_init_replay_buffer_max_length = config_transfer_baseline["planning_init_replay_buffer_max_length"]
    fake_steps_replay_buffer_max_length = config_transfer_baseline["fake_steps_replay_buffer_max_length"]
    num_eval_episodes = config_transfer_baseline["num_eval_episodes"]
    eval_interval = config_transfer_baseline["eval_interval"]
    log_interval = config_transfer_baseline["log_interval"]
    q_model_class = config_transfer_baseline["q_model"]["class"]
    q_model_params = config_transfer_baseline["q_model"]["params"]
    q_model_creator = lambda: R.get_q_model_constructor(q_model_class)(**q_model_params)
    compute_target_model = (baseline in [SOLUTION_1, SOLUTION_2,
                                         SOLUTION_3] and use_stop_transfer_mechanism) or baseline == DYNA_MODEL_LEARNING

    use_model_for_prediction = baseline in [SOLUTION_1, SOLUTION_2,
                                            SOLUTION_3, DYNA_MODEL_LEARNING,
                                            DYNA_RDM, DYNA_IMPERFECT, DYNA_AGGLO,
                                            DYNA_PERFECT, DYNA_IMPERFECT,
                                            DQN_EXTRA_UPDATES, DQN_EXTRA_STEPS]

    # tensorboard logs
    print("log_folder (transfer): tensorboard --logdir={}".format(str(pathlib.Path().absolute() / tf_log_path)))
    if run_tensorboard:
        import os
        os.system("tensorboard --logdir={} &".format(str(pathlib.Path().absolute() / tf_log_path)))
    writers = {}
    writers[baseline] = tf.summary.create_file_writer(str(tf_log_path / baseline))
    writers[baseline + "_transfer_model"] = tf.summary.create_file_writer(
        str(tf_log_path / (baseline + "_transfer_model")))
    writers[baseline + "_target_model"] = tf.summary.create_file_writer(str(tf_log_path / (baseline + "_target_model")))

    # use_target_q_network = config_baseline["use_target_q_network"]
    env_creator, _ = R.get_env(env_name)
    config_env = env_creator()
    print("config_env.horizon", config_env.horizon)
    wrapper_env = EnvWrapper(env_creator())  # , as_tensor=True)

    random_policy = RandomPolicy(config_env.action_space)

    # =============================
    # steps buffers
    # =============================

    target_steps = Memory(capacity=replay_buffer_max_length)
    extra_target_steps = Memory(capacity=replay_buffer_max_length)

    if compute_target_model:
        dyna_train_steps = Memory(class_transition=TransferTransition, capacity=replay_buffer_max_length)
        dyna_validation_steps = Memory(class_transition=TransferTransition, capacity=replay_buffer_max_length)

    if use_model_for_prediction:
        planning_steps = Memory(capacity=fake_steps_replay_buffer_max_length)
        planning_init_steps = Memory(capacity=planning_init_replay_buffer_max_length)

    agent = DQNAgent(
        action_space=config_env.action_space,
        optimizer=optimizer,
        gamma=gamma,
        update_network_frequency=update_network_frequency,
        model_creator=q_model_creator
    )

    egreedy_policy = agent.egreedy_policy
    greedy_policy = agent.greedy_policy

    egreedy_policy.set_eps(config_transfer_baseline["eps_start"])

    if compute_target_model:
        if baseline == DYNA_MODEL_LEARNING:
            ratio_train_validation = 1.
        else:
            ratio_train_validation = 0.75

    def collect_steps(n, policy):
        for i_n in range(n):
            step = wrapper_env.collect_step(policy)
            target_steps.push(*step)
            if use_model_for_prediction:
                planning_init_steps.push(*step)
            if compute_target_model:
                transfer_step = step[:-2] + (-1,)
                if np.random.random() < ratio_train_validation:
                    dyna_train_steps.push(*transfer_step)
                else:
                    dyna_validation_steps.push(*transfer_step)
                # print(i_n, target_steps.memory[-1])

    if initial_collect_steps > 0:
        collect_steps(initial_collect_steps, random_policy)
        draw(C, env_creator, target_steps,
             env_name + "_" + id_baseline + "_{}({})".format("initial_steps", initial_collect_steps))

    # Evaluate the agent's policy once before training.
    _, avg_return = rollouts(env_creator, egreedy_policy, num_eval_episodes,
                             multiprocessing=False)  # C.multiprocessing)  # , as_tensor=True)
    _, avg_return_greedy = rollouts(env_creator, greedy_policy, num_eval_episodes,
                                    multiprocessing=False)  # C.multiprocessing)  # , as_tensor=True)
    returns_egreedy = [avg_return]
    returns_greedy = [avg_return_greedy]
    print(returns_egreedy, returns_greedy)

    transfer_reward_suspended = False
    transfer_dynamic_suspended = False
    if compute_target_model:

        target_gan_config = C["transfer"]["target_gan_config"]

        if C["onehot_action"]:
            action_size = config_env.action_space.n
        else:
            action_size = get_gym_size(C,config_env.action_space)

        models_objects = {}
        for model_key in target_gan_config["models_to_learn"]:
            if model_key == "reward" and use_true_reward_prediction:
                raise Exception("You are learn a model that you wont use (reward model)")
            model_objects = {}
            models_objects[model_key] = model_objects
            common_params = {**target_gan_config}
            del common_params["models_to_learn"]
            specific_params = target_gan_config["models_to_learn"][model_key]
            # model_key_config = {**common_params, **specific_params}
            model_key_config = override_dictionary(common_params, specific_params)
            models_objects[model_key]["config"] = model_key_config
            models_objects[model_key]["config"]["model_key"] = model_key
            model_objects["D"] = discriminator(action_size=action_size,
                                               obs_size=get_gym_size(C,config_env.observation_space),
                                               gan_config=model_key_config)
            model_objects["G"] = generator(action_size=action_size,
                                           obs_size=get_gym_size(C,config_env.observation_space),
                                           gan_config=model_key_config)
            model_objects["G_opti"] = optimizer(**model_key_config["G_optimizer"])
            model_objects["D_opti"] = optimizer(**model_key_config["D_optimizer"])
            log_dynamic = path / "target_model" / "logs" / model_key
            if model_key == "reward":
                real_out_str = "r_"
            elif model_key == "dynamics":
                real_out_str = "s_"
            elif model_key == "done":
                real_out_str = "done"
            else:
                raise Exception()
            model_objects["plots"], model_objects["writers"] = create_plots_and_writers(real_out_str, log_dynamic)
            model_objects["real_out_str"] = real_out_str
            model_objects["lambda_"] = model_key_config.get("lambda_", None)
            model_objects["D_n_updates"] = model_key_config["D_n_updates"]
            model_objects["G_n_updates"] = model_key_config["G_n_updates"]
            model_objects["epoch"] = 0
            print("logs folder ({}): tensorboard --logdir={}".format(
                model_key,
                str(pathlib.Path().absolute() / log_dynamic)))

    i_real_or_sim_step = 0
    train_loss = None

    if action_selection_for_model_prediction == "random":
        policy_for_model = random_policy
    elif action_selection_for_model_prediction == "greedy":
        policy_for_model = greedy_policy
    elif action_selection_for_model_prediction == "egreedy":
        policy_for_model = egreedy_policy
    else:
        raise Exception()

    schedule = ExponentialDecay(config_transfer_baseline["eps_start"], config_transfer_baseline["lambda_decay"])
    if C["args"]["show_plots"]:
        schedule.plot(num_real_steps)

    total_training_updates = 0

    def train():
        if baseline in [DYNA_MODEL_LEARNING]:
            m = Memory()
            m.append_all(planning_steps)
            m.append_all(target_steps)
        elif baseline in [DYNA_AGGLO, DYNA_RDM, DYNA_IMPERFECT, DYNA_PERFECT, SOLUTION_1, SOLUTION_2,
                          SOLUTION_3]:
            m = Memory()  # can make is faster by sampling separatly from the 2 buffers (with rdm proportion)
            m.append_all(planning_steps)
            m.append_all(target_steps)

        elif baseline in [DQN, DQN_EXTRA_UPDATES]:
            m = target_steps
        elif baseline in [DQN_EXTRA_STEPS]:
            m = Memory()
            m.append_all(extra_target_steps)
            m.append_all(target_steps)

        else:
            raise Exception()

        if config_transfer_baseline["use_tf_agent"]:
            # TODO construct experience with minibatch
            minibatch = m.sample(mini_batch_size, tf_agent=True)
            train_loss = agent.train(minibatch).loss
        else:
            minibatch = m.sample(mini_batch_size)
            train_loss = agent.train(minibatch)
        return train_loss

    dynamic_accuracies = []
    reward_accuracies = []

    offset_step_collection = config_transfer_baseline["offset_step_collection"]

    for i_real_step in range(num_real_steps):

        egreedy_policy.set_eps(schedule.compute(i_real_step))
        # ==================================================
        # Collecting sample for Q learning and model learning, and model testing
        # ==================================================
        if i_real_step >= offset_step_collection:
            collect_steps(1, egreedy_policy)
        i_real_or_sim_step += 1

        # ==================================================
        # Perform a DQN gradient step
        # ==================================================
        if i_real_or_sim_step % training_interval == 0:
            train_loss = train()
            total_training_updates += 1

        # ==================================================
        # we learn the model
        # ==================================================
        if compute_target_model and i_real_step % update_model_interval == 0:
            if len(dyna_train_steps.memory) <= 0:
                print("<<< warning >>> no samples in dyna_train_steps")
            else:
                dyna_train_samples = []
                if C["onehot_action"]:
                    for s, a, r_, s_, i_env in dyna_train_steps.memory:
                        sss = TransferTransition(s, to_one_hot(config_env.action_space.n, a), r_, s_, i_env)
                        dyna_train_samples.append(sss)
                else:
                    dyna_train_samples = dyna_train_steps.memory
                train_dataset, _ = create_ds(dyna_train_samples,
                                             ratio_train_to_test=1.0,
                                             obs_to_np=C["obs_to_np"])  # probably very slow, should just add new data
                train_dataset.batch(len(train_dataset))
                dyna_validation_samples = []
                if C["onehot_action"]:
                    for s, a, r_, s_, i_env in dyna_validation_steps.memory:
                        sss = TransferTransition(s, to_one_hot(config_env.action_space.n, a), r_, s_, i_env)
                        dyna_validation_samples.append(sss)
                else:
                    dyna_validation_samples = dyna_validation_steps.memory
                if len(dyna_validation_samples) > 0:
                    validation_dataset, _ = create_ds(dyna_validation_samples,
                                                      ratio_train_to_test=1.0,
                                                      obs_to_np=C["obs_to_np"])  # probably very slow, should just add new data
                    validation_dataset.batch(len(validation_dataset))
                else:
                    validation_dataset = None
                    if baseline in [SOLUTION_1, SOLUTION_2, SOLUTION_3]:
                        print("<<< warning >>> validation is empty right now, train dataset is of size: {}".format(
                            len(dyna_train_samples)))

                for model_key, model_objects in models_objects.items():
                    perform_an_epoch(plots=model_objects["plots"],
                                     writers=model_objects["writers"],
                                     real_out_str=model_objects["real_out_str"],
                                     epoch=model_objects["epoch"],
                                     all_train_samples=train_dataset,
                                     all_tests_samples=validation_dataset,
                                     lambda_=model_objects["lambda_"],
                                     G=model_objects["G"],
                                     D=model_objects["D"],
                                     G_opti=model_objects["G_opti"],
                                     D_opti=model_objects["D_opti"],
                                     gan_config=model_objects["config"],
                                     D_n_updates=model_objects["D_n_updates"],
                                     G_n_updates=model_objects["G_n_updates"],
                                     obs_size=get_gym_size(C,config_env.observation_space),
                                     stop={}, callback=None, ner=None, verbose=True,
                                     interval_eval_epoch=log_target_model_performance_interval)
                    model_objects["epoch"] += 1

        # ==================================================
        # Update accuracy
        # ==================================================
        if i_real_step % update_accuracy_interval == 0:
            if baseline == SOLUTION_1:
                raise NotImplementedError()  # TODO pick source model
            elif baseline == SOLUTION_2:
                raise NotImplementedError()  # TODO pick best source, and foward super_gan with [0. 0. 0. 0.5 0.5 0. 0. ...]
            elif baseline == SOLUTION_3:
                last_samples = target_steps.memory[-update_accuracy_interval:]

                # TODO optimisation with batch processing
                batch_s = [sample.s for sample in last_samples]
                batch_r_ = [sample.r_ for sample in last_samples]
                batch_a = [to_one_hot(config_env.action_space.n, sample.a) for sample in last_samples]
                batch_s_ = [sample.s_ for sample in last_samples]

                inputs_dynamic = tf.data.Dataset.from_tensor_slices((batch_s, batch_a, batch_s_)).batch(
                    len(last_samples))
                if not use_true_reward_prediction:
                    inputs_reward = tf.data.Dataset.from_tensor_slices((batch_s, batch_a, batch_r_)).batch(
                        len(last_samples))

                for i, (s, a, s_) in enumerate(inputs_dynamic):
                    if i > 0:
                        raise Exception()
                    accs = gan_baseline.get_model("best", SUPER_GAN, "dynamics", "D", "all_envs")([s, a, s_])
                    if config_gan_baseline["factorize_fake_output"]:
                        accs, _ = accs
                    for acc in accs:
                        dynamic_accuracies.append(acc.numpy())

                if not use_true_reward_prediction:
                    for i, (s, a, r_) in enumerate(inputs_reward):
                        if i > 0:
                            raise Exception()
                        r_new = tf.reshape(r_, (-1, 1))
                        accs = gan_baseline.get_model("best", SUPER_GAN, "reward", "D", "all_envs")([s, a, r_new])
                        if config_gan_baseline["factorize_fake_output"]:
                            accs, _ = accs
                        for acc in accs:
                            reward_accuracies.append(acc.numpy())
                mean_dynamic_acc = np.mean(dynamic_accuracies, axis=0)

                if not use_true_reward_prediction:
                    mean_reward_acc = np.mean(reward_accuracies, axis=0)
                else:
                    mean_reward_acc = None
                #     # z= tf.cast(np.random.normal(-1.0, 1.0, size=[len(s), C["gan_config"]["z_size"]]), tf.float32)
                #     # l1_loss_acc_reduced = tf.reduce_mean(np.abs(tf.cast(s_, tf.float32) - tf.cast(exp.super_G_dynamics([s, a, z, tf.tile(tf.expand_dims(tf.reduce_mean(exp.super_D_dynamics([s, a, s_]), axis=0),axis=0), (len(s),1))]), tf.float32)), axis=0)
                #     # l1_loss_acc_not_reduced =tf.reduce_mean(np.abs(tf.cast(s_, tf.float32) - tf.cast(exp.super_G_dynamics([s, a, z, exp.super_D_dynamics([s, a, s_])]), tf.float32)), axis=0)

        # ==================================================
        # compute if the transfer should be suspended or not
        # ==================================================
        if use_stop_transfer_mechanism and i_real_step % check_stop_transfer_interval == 0 \
                and baseline in [SOLUTION_1, SOLUTION_2, SOLUTION_3] \
                and len(dyna_validation_steps.memory) > 0:
            validation_samples = []
            if C["onehot_action"]:
                for s, a, r_, s_, i_env in dyna_validation_steps.memory:
                    sss = TransferTransition(s, to_one_hot(config_env.action_space.n, a), r_, s_, i_env)
                    validation_samples.append(sss)
            else:
                validation_samples = dyna_validation_steps.memory
            ds, _ = create_ds(validation_samples,
                              ratio_train_to_test=1.0,
                              obs_to_np=C["obs_to_np"])
            ds.batch(len(ds))
            validation_err_target_model_dynamic = None
            validation_err_target_model_reward = None
            validation_err_transfer_model_dynamic = None
            validation_err_transfer_model_reward = None

            if baseline == SOLUTION_3:

                for ii, (s, a, r_, s_, i_env) in enumerate(ds):
                    if ii > 0:
                        raise Exception()
                    z = tf.cast(np.random.normal(-1.0, 1.0, size=[len(s), config_gan_baseline["z_size"]]), tf.float32)
                    tilled_acc_dynamic = tf.tile(tf.expand_dims(mean_dynamic_acc, axis=0), (len(s), 1))
                    tilled_acc_reward = tf.tile(tf.expand_dims(mean_reward_acc, axis=0), (len(s), 1))
                    fake_s_ = gan_baseline.get_model("best", SUPER_GAN, "dynamics", "G", "all_envs")(
                        [s, a, z, tilled_acc_dynamic])
                    fake_r_ = gan_baseline.get_model("best", SUPER_GAN, "reward", "G", "all_envs")(
                        [s, a, z, tilled_acc_reward])
                    fake_s_from_dyna_model = models_objects["dynamics"]["G"]([s, a, z])
                    fake_r_from_dyna_model = models_objects["reward"]["G"]([s, a, z])
                validation_err_transfer_model_dynamic = np.mean(np.abs(fake_s_.numpy() - s_.numpy()))
                validation_err_transfer_model_reward = np.mean(np.abs(fake_r_.numpy() - r_.numpy()))
                validation_err_target_model_dynamic = np.mean(np.abs(fake_s_from_dyna_model.numpy() - s_.numpy()))
                validation_err_target_model_reward = np.mean(np.abs(fake_r_from_dyna_model.numpy() - r_.numpy()))
            prevtransfer_dynamic_suspended = transfer_dynamic_suspended
            transfer_dynamic_suspended = validation_err_target_model_dynamic < validation_err_transfer_model_dynamic
            prevtransfer_reward_suspended = transfer_reward_suspended
            transfer_reward_suspended = validation_err_target_model_reward < validation_err_transfer_model_reward
            if transfer_dynamic_suspended != prevtransfer_dynamic_suspended:
                if transfer_dynamic_suspended:
                    print("using target dynamics now")
                else:
                    print("using transfered dynamics now")
            if transfer_reward_suspended != prevtransfer_reward_suspended:
                if transfer_reward_suspended:
                    print("using target reward now")
                else:
                    print("using transfered reward now")

            with writers[baseline].as_default():
                tf.summary.scalar("transfer_dynamic", not transfer_dynamic_suspended, step=i_real_step)
            with writers[baseline + "_transfer_model"].as_default():
                tf.summary.scalar("validation_err_reward", validation_err_transfer_model_reward, step=i_real_step)
                tf.summary.scalar("validation_err_dynamic", validation_err_transfer_model_dynamic, step=i_real_step)

            with writers[baseline + "_target_model"].as_default():
                tf.summary.scalar("validation_err_dynamic", validation_err_target_model_dynamic, step=i_real_step)
                tf.summary.scalar("validation_err_reward", validation_err_target_model_reward, step=i_real_step)

        # ==================================================
        # collect extra step from model, and learn Q from it
        # ==================================================
        if use_model_for_prediction:
            extra_steps = 0
            if baseline == DYNA_PERFECT:
                samples = extract_not_done_samples(n, planning_init_steps)
                steps, _ = rollouts(env_creator, policy_for_model, n, k, samples,
                                    multiprocessing=False)  # , as_tensor=True)
                planning_steps.append_all(steps)
                extra_steps += len(steps)
                for _ in range(extra_steps):
                    if i_real_or_sim_step % training_interval == 0:
                        train_loss = train()
                        total_training_updates += 1
                    i_real_or_sim_step += 1
            elif baseline == SOLUTION_3:
                if not use_true_reward_prediction:
                    if transfer_reward_suspended and use_stop_transfer_mechanism:
                        G_reward = models_objects["reward"]["G"]
                    else:
                        G_reward = gan_baseline.get_model("best", SUPER_GAN, "reward", "G", "all_envs")
                else:
                    G_reward = None

                if transfer_dynamic_suspended and use_stop_transfer_mechanism:
                    G_dynamic = models_objects["dynamics"]["G"]
                else:
                    G_dynamic = gan_baseline.get_model("best", SUPER_GAN, "dynamics", "G", "all_envs")

                if transfer_reward_suspended and use_stop_transfer_mechanism:
                    z_size = target_gan_config["z_size"]
                else:
                    z_size = config_gan_baseline["z_size"]

                samples = extract_not_done_samples(n, planning_init_steps)
                if len(samples) > 0:
                    steps = collect_steps_from_gan_model(
                        env_creator=env_creator,
                        k=k,
                        policy=policy_for_model,
                        init_samples=samples,
                        z_size=z_size,
                        acc_dynamic=mean_dynamic_acc,
                        acc_reward=mean_reward_acc,
                        G_dynamics=G_dynamic, G_reward=G_reward,
                        log_level=1,
                        use_true_done_to_stop_prediction=use_true_done_to_stop_prediction,
                        super_gan_dynamic=(not transfer_dynamic_suspended),
                        super_gan_reward=(not transfer_reward_suspended),
                        use_true_reward_prediction=use_true_reward_prediction)
                    planning_steps.append_all(steps)
                    extra_steps += len(steps)
                for _ in range(extra_steps):
                    if i_real_or_sim_step % training_interval == 0:
                        train_loss = train()
                        total_training_updates += 1
                    i_real_or_sim_step += 1
            elif baseline in [DYNA_MODEL_LEARNING, DYNA_IMPERFECT, DYNA_RDM, DYNA_AGGLO]:

                if baseline == DYNA_IMPERFECT:
                    G_dynamics = gan_baseline.get_model("best", CLASSIC, "dynamics", "G", env_name)
                    if not use_true_reward_prediction:
                        G_reward = gan_baseline.get_model("best", CLASSIC, "reward", "G", env_name)
                    else:
                        G_reward = None
                    z_size = config_gan_baseline["z_size"]
                elif baseline == DYNA_MODEL_LEARNING:
                    G_dynamics = models_objects["dynamics"]["G"]
                    if not use_true_reward_prediction:
                        G_reward = models_objects["reward"]["G"]
                    else:
                        G_reward = None
                    z_size = target_gan_config["z_size"]
                elif baseline == DYNA_RDM:
                    env_names = set()

                    for (perf, type_gan, type_model, agent, name_env), _ in gan_baseline.models.items():
                        if name_env != "all_envs":
                            env_names.add(name_env)

                    i_rdm_env = np.random.randint(0, len(env_names))
                    G_dynamics = gan_baseline.get_model("best", CLASSIC, "dynamics", "G", env_names[i_rdm_env])
                    if not use_true_reward_prediction:
                        G_reward = gan_baseline.get_model("best", CLASSIC, "reward", "G", env_names[i_rdm_env])
                    else:
                        G_reward = None
                    z_size = config_gan_baseline["z_size"]
                elif baseline == DYNA_AGGLO:
                    G_dynamics = gan_baseline.get_model("best", AGGLOMERATED_GAN, "dynamics", "G", "all_envs")
                    if not use_true_reward_prediction:
                        G_reward = gan_baseline.get_model("best", AGGLOMERATED_GAN, "reward", "G", "all_envs")
                    else:
                        G_reward = None
                    z_size = config_gan_baseline["z_size"]
                else:
                    raise Exception()

                samples = extract_not_done_samples(n, planning_init_steps)
                if len(samples) > 0:
                    steps = collect_steps_from_gan_model(
                        env_creator=env_creator,
                        k=k, policy=policy_for_model,
                        init_samples=samples,
                        z_size=z_size,
                        acc_dynamic=None, acc_reward=None,
                        G_reward=G_reward,
                        G_dynamics=G_dynamics,
                        log_level=1,
                        use_true_done_to_stop_prediction=use_true_done_to_stop_prediction,
                        super_gan_dynamic=False,
                        super_gan_reward=False,
                        use_true_reward_prediction=use_true_reward_prediction)
                    planning_steps.append_all(steps)
                    extra_steps += len(steps)
                for _ in range(extra_steps):
                    if i_real_or_sim_step % training_interval == 0:
                        train_loss = train()
                        total_training_updates += 1
                    i_real_or_sim_step += 1

            elif baseline == DQN_EXTRA_UPDATES:
                # no need to collect, we just need the extra updates that follow
                for _ in range(n):
                    for _ in range(k):
                        if i_real_or_sim_step % training_interval == 0:
                            train_loss = train()
                            total_training_updates += 1
                        i_real_or_sim_step += 1
            elif baseline == DQN_EXTRA_STEPS:
                # collect from real env, instead of model, it is a cheating baseline (upper bound)
                for _ in range(n * k):
                    step = wrapper_env.collect_step(egreedy_policy)
                    extra_target_steps.push(*step)
                    if i_real_or_sim_step % training_interval == 0:
                        train_loss = train()
                        total_training_updates += 1
                    i_real_or_sim_step += 1
            else:
                raise Exception()

            # Perform a DQN gradient step (with simulated and real samples) for n*k update
            # (we should do it directly after collect
            # but it is faster to first collect with parrallelism, then train

        # ====================================================
        # evaluation
        # ====================================================

        if i_real_step % log_interval == 0:
            strg = "i_real_step= {}(+{}) training_update={}: loss = {}"
            args_str = (i_real_step, initial_collect_steps, total_training_updates, train_loss)
            if baseline == SOLUTION_3:
                strg += " dynamic_acc={} reward_acc={}"
                args_str += (np.mean(dynamic_accuracies, axis=0), np.mean(reward_accuracies, axis=0))
            print(strg.format(*args_str))

        if i_real_step % eval_interval == 0:
            for policy, returns, writer_str in [(greedy_policy, returns_greedy, "greedy"),
                                                (egreedy_policy, returns_egreedy, "egreedy")]:
                m, avg_return = rollouts(env_creator, policy, num_eval_episodes, multiprocessing=False)
                print('i_real_step= {}(+{}) training_update={} return ({}) = {}'
                      .format(i_real_step, initial_collect_steps, total_training_updates, str(policy), avg_return))
                returns.append(avg_return)
                with writers[baseline].as_default():
                    tf.summary.scalar(writer_str, avg_return, step=i_real_step)
                draw(C, env_creator, m, env_name + "_" + id_baseline + "_{}_{}".format(writer_str, i_real_step))

            do_break = False
            if len(returns_greedy) > stop_mean_return[0]:
                do_break = True
                for xx in range(1, stop_mean_return[0] + 1):
                    do_break = do_break and returns_greedy[-xx] >= stop_mean_return[1]
            if do_break:
                break
        if i_real_step % plot_interval == 0:
            plot_return(baseline, path, eval_interval, returns_egreedy, returns_greedy,
                        initial_collect_steps, C.show_plots)


def draw(C, env_creator, memory, title):
    test_env = env_creator()
    if isinstance(test_env, EnvGridWorld):
        from rltl.envs.gridworld.world import World
        world = World(test_env)
        world.draw_frame()
        world.draw_cases()
        world.draw_lattice()
        for sample in memory.memory:
            s = [None, None]
            s_ = [None, None]
            if test_env.normalise_state:
                s[0] = sample[0][0] * test_env.w
                s[1] = sample[0][1] * test_env.h
                s_[0] = sample[3][0] * test_env.w
                s_[1] = sample[3][1] * test_env.h
            world.draw_trajectory([(s, None, None, s_, None, None)], line_width=1, rgba=(0.5, 0.5, 1, 1))
        world.save(
            str(C.path / "transfer" / "worlds" / title))
        if C.show_plots:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            img = mpimg.imread(str(
                C.path / "transfer" / "worlds" / (title + ".png")))
            imgplot = plt.imshow(img)
            plt.show()


def plot_return(title, folder, eval_interval, returns_egreedy, returns_greedy, initial_collect_steps, show_plot=False):
    xy_egreedy = np.array(
        [np.array(range(len(returns_egreedy))) * eval_interval + initial_collect_steps, returns_egreedy])
    xy_greedy = np.array([np.array(range(len(returns_greedy))) * eval_interval + initial_collect_steps, returns_greedy])

    plt.plot(xy_egreedy[0], xy_egreedy[1], label="returns_egreedy")
    plt.plot(xy_greedy[0], xy_greedy[1], label="returns_greedy")
    plt.title(title + "(initstep: {})".format(initial_collect_steps))
    plt.legend()
    plt.ylabel('Average Return')
    plt.xlabel('steps')
    plt.ylim(top=1.2, bottom=-1.2)

    np.save(str(folder) + "/" + "egreedy.npy", xy_egreedy)
    np.save(str(folder) + "/" + "greedy.npy", xy_greedy)

    plt.savefig(str(folder) + "/{}.png".format(title))
    if show_plot:
        plt.show()
    plt.close()
