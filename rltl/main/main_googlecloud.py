command= "bash main.sh " \
         "configs/nicolas/main_generalisation.yaml " \
         "configs/nicolas/google_cloud.yaml " \
         "/home/nicolas_carrara1u/anaconda3/bin/python " \
         "/home/nicolas_carrara1u/work/rltl"

command= "PYTHONPATH=\"/home/nicolas_carrara1u/rltl\" " \
         "/home/nicolas_carrara1u/anaconda3/bin/python main.py " \
         "configs/nicolas/main_generalisation.yaml " \
         "--override_config_file configs/nicolas/google_cloud.yaml " \
         "--jobs learn_gans " \
         "--gan_baselines hyper_gan_0 " \

from subprocess import Popen, PIPE, CalledProcessError

with Popen(command, stdout=PIPE,shell=True, bufsize=1, universal_newlines=True) as p:
    for line in p.stdout:
        print(line, end='') # process line here

if p.returncode != 0:
    raise CalledProcessError(p.returncode, p.args)