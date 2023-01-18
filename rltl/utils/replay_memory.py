import random

import sys
import logging
from copy import copy

#from tf_agents.trajectories.time_step import StepType
#from tf_agents.trajectories.trajectory import Trajectory

from rltl.utils.transition import TransitionGym
from rltl.utils.utils_os import save_object, load_object
import tensorflow as tf


class Memory(object):
    logger = logging.getLogger(__name__)

    def __init__(self, capacity=sys.maxsize, class_transition=TransitionGym):
        self.capacity = capacity
        self.class_transition = class_transition
        # if self.store_as_tensor:
        # self.s = tf.TensorArray(tf.float32, size=capacity, dynamic_size=True, clear_after_read=False)
        # self.a = tf.TensorArray(tf.float32, size=capacity, dynamic_size=True, clear_after_read=False)
        # self.s_ = tf.TensorArray(tf.float32, size=capacity, dynamic_size=True, clear_after_read=False)
        # self.r_ = tf.TensorArray(tf.float32, size=capacity, dynamic_size=True, clear_after_read=False)
        # self.done = tf.TensorArray(tf.float32, size=capacity, dynamic_size=True, clear_after_read=False)
        # else:
        self.memory = []

        self.position = 0

    def reset(self):
        self.memory = []
        self.position = 0

    def split(self, sizes):
        samples = self.sample(len(self))
        memories = []
        prev_size = 0
        csizes = copy(sizes)
        csizes.append(1)
        for size in csizes:
            m = Memory()
            m.append_all(samples[prev_size:size])
            prev_size = size
            memories.append(m)

    def append_all(self, memory):
        for sample in memory.memory:
            self.push(*sample)

    def copy(self):
        m = Memory()
        m.append_all(self)
        return m

    def push(self, *args):

        #     self.s.write(self.position, args[0])
        #     self.a.write(self.position, args[1])
        #     self.r_.write(self.position, args[2])
        #     self.s_.write(self.position, args[3])
        #     self.done.write(self.position, args[4])
        # else:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.class_transition(*args)



        # if self.gpu is not None:
        #     with tf.device(self.gpu.name):
        #         self.memory[self.position].s = tf.Tensor(self.memory[self.position].s)
        #         self.memory[self.position].s_ = tf.Tensor(self.memory[self.position].s_)
        #         self.memory[self.position].a = tf.Tensor(self.memory[self.position].a)
        #         self.memory[self.position].r_ = tf.Tensor(self.memory[self.position].r_)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, tf_agent=False):
        # if batch_size > len(self.memory):
        #     print("<<<WARNING, you are sampling {} samples but you only have {} samples>>>".format(batch_size,
        #                                                                                            len(self.memory)))
        batch_size = len(self.memory) if batch_size > len(self.memory) else batch_size

 #       if tf_agent:
 #           # traj = Trajectory()
 #           indexes = np.random.choice(list(range(len(self.memory))), batch_size)
#
#            # res = [(i, self.memory[i]) for i in idx]
#            # return res
#
#            step_type = np.ones(shape=(batch_size, 2))  # all MID
#            observation = np.zeros(shape=(batch_size, 2) + self.memory[0].state.shape)
#            action = np.zeros(shape=(batch_size, 2))
#            next_step_type = np.ones(shape=(batch_size, 2))  # all MID, unless DONE
#            reward = np.zeros(shape=(batch_size, 2))
#            discount = np.ones(shape=(batch_size, 2))

#            for i,idx in enumerate(indexes):
#                step = self.memory[idx]
#                observation[i][0] = step.s
#                observation[i][1] = step.s_
#                action[i] = step.a
#                reward[i] = step.r_
#
#                if step.done:
#                    next_step_type[i][1] = StepType.LAST
#                    discount[i][1] = 0


 #           t = Trajectory(step_type=tf.constant(step_type,dtype=tf.int32),
 #                          observation=tf.constant(observation,dtype=tf.float32),
 #                          action=tf.constant(action,dtype=tf.int64),
 #                          policy_info=(),
  #                         next_step_type=tf.constant(next_step_type,dtype=tf.int32),
   #                        reward=tf.constant(reward,dtype=tf.float32),
    #                       discount=tf.constant(discount,dtype=tf.float32))
    #        return t
#
 #       else:
        return random.sample(self.memory, batch_size)

    def save(self, path, indent=0):
        memory = [t._asdict() for t in self.memory]
        save_object(memory, path, indent)

    @staticmethod
    def static_load(path):
        m = Memory()
        m.load(path)
        return m

    def load(self, path):
        self.logger.info("loading memory at {}".format(path))
        memory = load_object(path)
        self.reset()
        for idata, data in enumerate(memory):
            self.push(*data.values())

    # def apply_feature_to_states(self, feature):
    #     Memory.logger.info('Applying feature to states in memory')
    #     for i in range(len(self.memory)):
    #         state, action, reward, next_state, done, info = self.memory[i]
    #         self.memory[i] = TransitionGym(feature(state), action, reward, feature(next_state), done, info)

    # def _tensor_to_list(self, t):
    #     return t.squeeze().cpu().numpy().tolist()

    # def to_lists(self):
    #     Memory.logger.info('Transforming memory to list')
    #     for i in range(len(self.memory)):
    #
    #         state, action, reward, next_state, done, info = self.memory[i]
    #         state = self._tensor_to_list(state)
    #         if not done:
    #             next_state = self._tensor_to_list(next_state)
    #         else:
    #             next_state = None
    #         action = self._tensor_to_list(action)
    #         reward = self._tensor_to_list(reward)
    #         self.memory[i] = TransitionGym(state, action, reward, next_state, done, info)
    #
    #     return self

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        rez = "".join(["{:05d} {} \n".format(it, str(t)) for it, t in enumerate(self.memory)])
        return rez[:-1]


import numpy as np

# m = Memory(capacity=10, store_as_tensor=True)
# s = np.array([1, 2, 3, 4])
# a = np.array([1])
# s_ = np.array([9, 10, 11, 12])
# r_ = np.array([99])
# done = np.array([False])
# m.push(s, a, r_, s_,done)
# print("hello")
# m = ReplayMemory(1000)
# m.push([1, 2, 3], 4, 18, [5, 6, 7])
# m.push([1, 2, 3], 4, 18, [5, 6, 7])
# m.push([1, 2, 3], 4, 18, [5, 6, 7])
# m.dump_memory("memory.json")
# # import json
# # print json.dumps(m)
# m = ReplayMemory(100)
# m.load_memory("memory.json")
# print m.memory
