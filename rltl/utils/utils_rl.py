import base64
# from multiprocessing import Pool

import IPython
import imageio
import numpy as np
from rltl.main.policy import RandomPolicy

from rltl.main.policy import SingleActionPolicy
from rltl.utils.registry import R
from rltl.utils.replay_memory import Memory
from rltl.utils.transition import TransitionGym
import tensorflow as tf
import gym


def to_one_hot(n, a):
    onehot_a = np.zeros(n)
    onehot_a[a] = 1
    return onehot_a


def compute_combinaison(values):
    res = []
    for v in values[0]:
        if len(values) > 1:
            sub_lists = compute_combinaison(values[1:])
            for sub_list in sub_lists:
                res.append([v] + sub_list)
        else:
            res.append([v])
    return res


def exploration(env_creator, type, one_hot_action,
                change_env_init_state, policy = None,
                n_episodes=None, n_samples=None, deltas=None,
                std=0.0, n_repeat=1):
    test_env = env_creator()
    if policy is None:
        policy = RandomPolicy(action_space=test_env.action_space)
    else:
        policy = R.d["policy"][policy]
    if type == "random":
        m, _ = rollouts(
            env_creator,
            policy,
            num_episodes=n_episodes,
            init_samples=None,
            onehot_action=one_hot_action,
            multiprocessing=False)
    elif type == "uniform":
        m = Memory()
        for _ in range(n_samples):
            s = test_env.observation_space.sample()
            samples, _ = rollout(
                env_creator,
                policy,
                k=1,
                init_sample=TransitionGym(s, None, None, None, None, None),
                onehot_action=one_hot_action,
                change_env_init_state=change_env_init_state)
            m.append_all(samples)
    elif type == "grid":
        space = test_env.observation_space
        m = Memory()

        # x = space.low
        # values = [x]
        # while x < space.high:
        #     x += delta
        #     values.append(x)

        values = []
        for feat in range(space.shape[0]):
            x = space.low[feat]
            vs = [x]
            while x < space.high[feat]:
                x += deltas[feat]
                vs.append(x)
            values.append(vs)

        init_states = [np.array(x) for x in compute_combinaison(values)]
        if n_repeat < 1:
            n_repeat = 1
            print("[utils_rl] Warning n_samples_by_dot was < 1")
        for s in init_states:
            for _ in range(n_repeat):
                # noisy randomise state
                s_noisy = np.zeros(s.shape)
                for i in range(len(s)):
                    s_noisy[i] = np.random.normal(s[i],
                                                  std)  # s[i] - deltas[i] / 10 + np.random.random(1) * 2 * deltas[i] / 10
                    s_noisy[i] = space.low[i] if s_noisy[i] < space.low[i] else (
                        space.high[i] if s_noisy[i] > space.high[i] else s_noisy[i])
                for a in range(test_env.action_space.n):
                    samples, _ = rollout(
                        env_creator,
                        SingleActionPolicy(a=a),
                        k=1,
                        init_sample=TransitionGym(s_noisy, None, None, None, None, None),
                        onehot_action=one_hot_action,
                        change_env_init_state=change_env_init_state)
                    m.append_all(samples)
    elif type == "old_grid":
        space = test_env.observation_space
        m = Memory()

        # x = space.low
        # values = [x]
        # while x < space.high:
        #     x += delta
        #     values.append(x)

        values = []
        for feat in range(space.shape[0]):
            x = space.low[feat]
            vs = [x]
            while x < space.high[feat]:
                x += deltas[feat]
                vs.append(x)
            values.append(vs)

        init_states = [np.array(x) for x in compute_combinaison(values)]
        n_samples_by_dot = int(n_samples / len(init_states))
        for s in init_states:
            for _ in range(n_samples_by_dot):
                # noisy randomise state
                s_noisy = np.zeros(s.shape)
                for i in range(len(s)):
                    s_noisy[i] = np.random.normal(s[i],
                                                  std)  # s[i] - deltas[i] / 10 + np.random.random(1) * 2 * deltas[i] / 10
                    s_noisy[i] = space.low[i] if s_noisy[i] < space.low[i] else (
                        space.high[i] if s_noisy[i] > space.high[i] else s_noisy[i])
                samples, _ = rollout(
                    env_creator,
                    RandomPolicy(action_space=test_env.action_space),
                    k=1,
                    init_sample=TransitionGym(s_noisy, None, None, None, None, None),
                    onehot_action=one_hot_action,
                    change_env_init_state=change_env_init_state)
                m.append_all(samples)

    else:
        raise Exception()

    return m


def change_env_state_gridworld(env, state):
    true_s = np.array(state)
    if env.normalise_state:
        true_s[0] = true_s[0] * env.w
        true_s[1] = true_s[1] * env.h
    env.state = true_s


def change_env_state(env, state):
    env.state = state


def reverse_one_hot(a):
    a = np.where(a == 1)
    return a[0]


def rollouts(env_creator,
             policy,
             num_episodes=10,
             k=None,
             init_samples=None,
             onehot_action=False,
             multiprocessing=False):
    if init_samples is None:
        init_samples = [None] * num_episodes
    else:
        if len(init_samples) < num_episodes:
            print("Warning, not enough init states ({}) for {} episodes".format(len(init_samples), num_episodes))
            num_episodes = len(init_samples)
    args_pool = [
        (env_creator, policy, k, init_samples[i_episode] if init_samples is not None else None, onehot_action) for
        i_episode in
        range(num_episodes)]  # , as_tensor) for i_episode in range(num_episodes)]
    if multiprocessing:
        from ray.util.multiprocessing import Pool
        # from multiprocessing import Pool

        with Pool(len(args_pool)) as p:
            ret = p.starmap(rollout, args_pool)
    else:
        ret = []
        for arg in args_pool:
            ret.append(rollout(*arg))

    mean_episode_reward = np.mean([rew for _, rew in ret])

    m = Memory()
    for episode_memory, _ in ret:
        m.append_all(episode_memory)

    return m, mean_episode_reward


def rollout(env_creator, policy, k=None, init_sample=None, onehot_action=False, change_env_init_state=None):
    env = env_creator()
    m = Memory()
    if change_env_init_state is None:
        change_env_init_state = change_env_state
    if init_sample is not None:
        s = init_sample.s
        env.reset()
        change_env_init_state(env, s)
        done = init_sample.done
        if done:
            raise Exception("a sample has a done")
    else:
        s = env.reset()
        done = False
    episode_reward = 0.0
    i_k = 0
    while not done:
        a = policy.act(s, batch_mode=False)
        s_, r_, env_done, info = env.step(a)
        episode_reward += r_
        # if not as_tensor:
        if onehot_action and isinstance(env.action_space, gym.spaces.Discrete):
            a = to_one_hot(env.action_space.n, a)
        m.push(s, a, r_, s_, env_done, info)
        s = s_
        i_k += 1
        done = env_done or (k is not None and i_k >= k)
    return m, episode_reward


class EnvWrapper():
    def __init__(self, env, onehot_action=False):  # , as_tensor=True):
        self.env = env
        self.done = True
        self.s = None
        self.onehot_action = onehot_action
        # self.as_tensor = as_tensor
        self.i = None

    def collect_step(self, policy):
        if self.done:
            self.s = self.env.reset()
            self.i = 0
            self.done = False
        s = self.s
        a = policy.act(s, batch_mode=False)
        s_, r_, done, info = self.env.step(a)

        if self.onehot_action and isinstance(self.env.action_space, gym.spaces.Discrete):
            a = to_one_hot(self.env.action_space.n, a)
        # if self.as_tensor is None:
        self.s = s_
        self.i += 1
        # self.done = done or (self.env.horizon is not None and self.i >= self.env.horizon)
        ret = s, a, r_, s_, done, info
        self.done = done
        return ret

    def collect_steps(self, n_steps, policy):
        steps = Memory()
        for _ in range(n_steps):
            steps.push(*self.collect_step(policy))
        return steps


def collect_steps_from_gan_model(env_creator,
                                 k, policy,
                                 init_samples,
                                 z_size,
                                 acc_dynamic, acc_reward,
                                 G_dynamics,
                                 G_reward,

                                 use_true_done_to_stop_prediction=False,
                                 log_level=0, super_gan_dynamic=True, super_gan_reward=True,
                                 use_true_reward_prediction=False):
    n = len(init_samples)
    if n <= 0:
        raise Exception("init_samples is empty")
    if log_level > 1:
        if super_gan_reward:
            print("using reward acc: {}".format(acc_reward))
        if super_gan_dynamic:
            print("using dynamic acc: {}".format(acc_dynamic))
    if super_gan_dynamic:
        tilled_acc_dynamic = tf.tile(tf.expand_dims(acc_dynamic, axis=0), (n, 1))
    if super_gan_reward and not use_true_reward_prediction:
        tilled_acc_reward = tf.tile(tf.expand_dims(acc_reward, axis=0), (n, 1))
    else:
        tilled_acc_reward = None
    s = [sample.s for sample in init_samples]
    m = Memory()
    if log_level > 0:
        all_l1_err = []
        all_l1_err_reward = []
        reconstruction_error_wrt_k = [[] for _ in range(k)]
        # reconstruction_error_wrt_n = [[] for _ in range(n)]
    for ik in range(k):
        a = policy.act(s, batch_mode=True)
        a = [to_one_hot(env_creator().action_space.n, a) for a in a]
        sa = tf.data.Dataset.from_tensor_slices((s, a)).batch(n)
        dones_mask = np.zeros((len(s)))
        for i, (s, a) in enumerate(sa):
            if i > 0:
                raise Exception
            z = tf.cast(np.random.normal(-1.0, 1.0, size=[len(s), z_size]), tf.float32)

            input_dynamics = [s, a, z]
            if super_gan_dynamic:
                input_dynamics.append(tilled_acc_dynamic)
            input_rewards = [s, a, z]
            if super_gan_reward and not use_true_reward_prediction:
                input_rewards.append(tilled_acc_reward)

            s_ = G_dynamics(input_dynamics)
            if not use_true_reward_prediction:
                r_ = G_reward(input_rewards)
            for i_n in range(len(s)):
                # TODO, maybe dont add False at done ?
                si = s[i_n].numpy()
                ai = reverse_one_hot(a[i_n].numpy()).squeeze()
                si_ = s_[i_n].numpy()

                if log_level > 0 or use_true_done_to_stop_prediction or use_true_reward_prediction:
                    mem = rollout(env_creator=env_creator,
                                  k=1,
                                  policy=SingleActionPolicy(ai),
                                  init_sample=TransitionGym(si, None, None, None, None, None))[0]
                    step = mem.memory[-1]

                if not use_true_reward_prediction:
                    ri_ = r_[i_n].numpy().squeeze()
                else:
                    ri_ = step.r_

                if log_level > 0:
                    recons_err = np.abs(si_ - step.s_)
                    reconstruction_error_wrt_k[ik].append(recons_err)
                    # reconstruction_error_wrt_n[i_n].append(recons_err)
                    all_l1_err.append(recons_err)
                    if not use_true_reward_prediction:
                        all_l1_err_reward.append(np.abs(ri_ - step.r_))

                if log_level > 1:
                    print("n={} k={} s={} fake_s_ ={}=?{}=s_ fake_r_={:.2f}?={:.2f}=r_  --> L1= {}"
                          .format(i_n, ik, si, si_, step.s_, ri_, step.r_,
                                  np.abs(si_ - step.s_)))
                if use_true_done_to_stop_prediction:
                    done = step.done  # TODO, remove cheat code here
                else:
                    done = False
                sample = si, ai, ri_, si_, done, None
                m.push(*sample)
                dones_mask[i_n] = done
                # if done:
                #     print("done")
            if use_true_done_to_stop_prediction:
                s = np.compress(1 - dones_mask, s_, axis=0)
                if super_gan_dynamic:
                    tilled_acc_dynamic = np.compress(1 - dones_mask, tilled_acc_dynamic, axis=0)
                if super_gan_reward and not use_true_reward_prediction:
                    tilled_acc_reward = np.compress(1 - dones_mask, tilled_acc_reward, axis=0)
                # dont predict if true env returned done (not that using true env is cheating, need to predict done)
            else:
                s = s_
    if log_level > 1:
        print("reconstruction errors from fake rollouts:")
        print("--- wrt k")
        for i, err in enumerate(reconstruction_error_wrt_k):
            print("k={} -> l1={}".format(i, np.mean(err, axis=0)))

        # print("--- wrt n")
        # for i, err in enumerate(reconstruction_error_wrt_n):
        #     print("n={} -> l1={}".format(i, np.mean(err, axis=0)))
    if log_level > 0:
        print("l1={} (dynamic) l1={} (reward)".format(np.mean(all_l1_err, axis=0), np.mean(all_l1_err_reward, axis=0)))

    return m


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


def create_policy_eval_video(policy, filename, eval_env, eval_py_env, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())
    return embed_mp4(filename)


def extract_not_done_samples(n, memory):
    all_samples = memory.sample(len(memory))
    samples = []
    max_len = np.min((len(all_samples), n))
    i = 0
    while len(samples) < max_len and i < len(all_samples):
        sample = all_samples[i]
        if not sample.done:
            samples.append(sample)
        i += 1
    return samples
