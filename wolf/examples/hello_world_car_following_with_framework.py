from wolf.ray.main import runs
import os
import time
import yaml

# runs(config_file_path="wolf/ray/tests/traffic_env/car_following/random.yaml",
#      override_config_file_path="wolf/ray/tests/override_configs/yifei.yaml")

def generate_tmp_config_path(config_file_path):
     with open(config_file_path, 'r') as f:
          config = yaml.full_load(f)

     experiment_name = list(config['ray']['run_experiments']['experiments'].keys())[0]
     env_config = config['ray']['run_experiments']['experiments'][experiment_name]['config']['env_config']

     reward_folder = f'reward_analysis_{int(time.time())}'
     try:
          record_path = os.environ['SOW_RECORD_PATH']
     except KeyError:
          print("[Warning]: Assign Variable SOW_RECORD_PATH to enabel record")
          return
     reward_analysis_dir_path = os.path.join(record_path, reward_folder)
     if not os.path.exists(reward_analysis_dir_path):
          os.makedirs(reward_analysis_dir_path)

     env_config['reward_folder'] = reward_analysis_dir_path

     # Make a username dir in tmp folder     
     if not os.path.exists('/tmp/yifei'):
          os.mkdir('/tmp/yifei')
     
     # GEnerate a temp config file
     tmp_config_file_path = f'/tmp/yifei/ddpg_{int(time.time())}.yaml'
     with open(tmp_config_file_path, 'w') as f:
          yaml.dump(config, f)
     return tmp_config_file_path

config_file_path = "wolf/ray/tests/traffic_env/car_following/ddpg.yaml"

tmp_config_file_path = generate_tmp_config_path(config_file_path)

runs(config_file_path=tmp_config_file_path,
     override_config_file_path="wolf/ray/tests/override_configs/yifei.yaml")

# try:
#      record_path = os.environ['SOW_RECORD_PATH']
# except KeyError:
#      record_path = '/home/aiyifei/Study/rl_project/records'

# reward_analysis_dir_path = os.path.join(record_path, f'reward_analysis')
# new_reward_analysis_dir_path = os.path.join(record_path, f'reward_analysis_{int(time.time())}')
# if os.path.exists(reward_analysis_dir_path):
#      os.rename(reward_analysis_dir_path, new_reward_analysis_dir_path)
# runs(config_file_path="../ray/tests/traffic_env/car_following/ddpg.yaml",
#      override_config_file_path="../ray/tests/override_configs/yifei.yaml")