# =============================================================================
# 2020.03.25 Updated. ray-1.2.0 now available.

# In order to use the newest WOLF on CC, please set up a new virtualenv with
# python/3.8 as follow
# =============================================================================


# install SUMO package
# TODO: using SUMO-1.8.0 rather than 1.6.0
mkdir downloads
cd downloads
curl -O https://sumo.dlr.de/releases/1.6.0/sumo-src-1.6.0.tar.gz
tar xzvf sumo-src-1.6.0.tar.gz
mv -r sumo-1.6.0 ~/sumo-1.6.0
cd ~/sumo-1.6.0
cmake .
make
echo "export SUMO_HOME=~/sumo-1.6.0" >> ~/.bashrc
echo "export PATH=$HOME/sumo-1.6.0/bin:$PATH" >> ~/.bashrc
source ~/.bashrc


# download code
mkdir ~/projects/def-ssanner/<user-name>/repos
cd ~/projects/def-ssanner/<user-name>/repos
git clone https://github.com/parthjaggi/flow.git
git clone http://116.66.187.35:4502/gitlab/its/sow45_code.git


# set up the virtualenv
### load python-3.8 BEFORE activating any virtualenv
module load python/3.8

### create & activate virtual env
mkdir ~/env
cd ~/env
virtualenv --no-download ./wolf
source ~/env/wolf/bin/activate
pip install --no-index -U pip

### install ray-1.2.0
# pip install -U ray  # deprecated, this only gives you ray-1.1.0
cd ~/downloads
curl -O https://files.pythonhosted.org/packages/8b/eb/c4b2b01e4b7f86a5c2e7d34b33d7c128ae8dd13dbb4a2552374f0eeceaa4/ray-1.2.0-cp38-cp38-manylinux2014_x86_64.whl
cp ray-1.2.0-cp38-cp38-manylinux2014_x86_64.whl ray-1.2.0-cp38-cp38-linux_x86_64.whl
pip install ray-1.2.0-cp38-cp38-linux_x86_64.whl

### install ray[rllib]
##### resolve some ray[rllib]'s dependancies manually
pip install lz4
cd ~/downloads
curl -O https://files.pythonhosted.org/packages/9c/6f/220c45977e6f85cbe63cd978c5cb774aa7c71ef9fb52b45f69c2611af010/opencv_python-4.1.2.30-cp38-cp38-manylinux1_x86_64.whl
curl -O https://files.pythonhosted.org/packages/8e/7f/671c15a2bf4701a034c04425193e6cd33d677852db17b18bc6320ffa63bd/opencv_python_headless-4.2.0.32-cp38-cp38-manylinux1_x86_64.whl
cp opencv_python-4.1.2.30-cp38-cp38-manylinux1_x86_64.whl opencv_python-4.1.2.30-cp38-cp38-linux_x86_64.whl
cp opencv_python_headless-4.2.0.32-cp38-cp38-manylinux1_x86_64.whl opencv_python_headless-4.2.0.32-cp38-cp38-linux_x86_64.whl
pip install opencv_python-4.1.2.30-cp38-cp38-linux_x86_64.whl opencv_python_headless-4.2.0.32-cp38-cp38-linux_x86_64.whl
##### install it
pip install --no-index ray[rllib]

### install flow & wolf
cd flow
pip install --no-index -e .
cd ../sow45_code
pip install --no-index -e .

### extra packages
pip install --no-index matplotlib torch imutils
pip install --no-index tensorflow_gpu
