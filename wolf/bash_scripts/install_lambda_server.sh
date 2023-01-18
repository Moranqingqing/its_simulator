# =============================================================================

# install conda for python=3.8
# https://docs.conda.io/en/latest/miniconda.html

# download code
cd ~
mkdir repos
cd repos
### clone parth/FLOW
git clone https://github.com/parthjaggi/flow.git
### clone WOLF
git clone http://116.66.187.35:4502/gitlab/its/sow45_code.git


# config environment
conda create -n wolf python=3.8 matplotlib
conda activate wolf
### install cuda 11.0
conda install cudatoolkit=11.0
### install cudnn 8.1
cd ~
cp /home/admin/wheels/cudnn-11.2-linux-x64-v8.1.1.33.tgz .
tar xzvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
cp cuda/lib64/* ~/anaconda3/envs/wolf/lib/  # replace anaconda3 with miniconda3 if using that.
rm cudnn-11.2-linux-x64-v8.1.1.33.tgz
rm -r cuda/
### install tf 2.4
pip install tensorflow-gpu==2.4
### other packages, clone flow & wolf in advance
pip install -U ray
pip install ray[rllib]
cd ~/repos/flow
pip install -e .
cd ~/repos/sow45_code
pip install -e .
pip install torch imutils
