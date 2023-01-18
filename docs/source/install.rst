Installation
============

To get wolf running, you need few things: wolf, SUMO (a traffic simulator), flow (a traffic simulation framework), and ray (a framework for building distributed ML applications) together with RLlib (a reinforcement learning library). In this documentation, we will lead you going through all the steps to make every component ready.

Once each component is installed successfully, you might get some missing module bugs from Python. Just install the missing module using your OS-specific package manager / installation tool. Follow the shell commands below to get started.

Wolf is designed to work with multiple traffic simulators: SUMO, AIMSUN, CTM (built-in). Wolf + SUMO is fully tested and ready to use now. Wolf is compatible with AIMSUN, but not fully tested yet.



Installing SUMO
~~~~~~~~~~~~~~~

There are two ways to install SUMO. You can either install the latest released version of SUMO by using :code:`sudo apt-get`, or locally install SUMO by compiling from the source code. For more details, you can checkout the SUMO's documentation: https://sumo.dlr.de/docs/Downloads.php.

**Add SUMO to your system**

If you are free to use the :code:`sudo` command, we recommand to install SUMO with the first method, which is easier. You can add the most recent SUMO to your Ubuntu system by doing these:

.. code-block:: sh

    sudo add-apt-repository ppa:sumo/stable
    sudo apt-get update
    sudo apt-get install sumo sumo-tools sumo-doc

Then you have to add SUMO to your system path. Please add the following two lines into your :code:`.bashrc` file.

.. code-block:: sh

    export SUMO_HOME=/usr/share/sumo
    export PATH=/usr/share/sumo/bin:$PATH

And source the :code:`.bashrc` file.

.. code-block:: sh

    source ~/.bashrc

**Compile SUMO from source code**

The other option for installing SUMO without :code:`sudo` is that compile SUMO from the source code locally. We recommand you to install version 1.6.0 of SUMO which is fully tested and works well.

.. code-block:: sh
    
    # download the code, unzip it, and put the folder to the place you want.
    curl -O https://sumo.dlr.de/releases/1.6.0/sumo-src-1.6.0.tar.gz
    tar xzvf sumo-src-1.6.0.tar.gz
    mv -r sumo-1.6.0 /the/path/to/sumo

    # compile the code
    cd /the/path/to/sumo
    cmake .
    make

You still need to add SUMO to your system path following the previous introduction. Please add the following two lines into your :code:`.bashrc` file.

.. code-block:: sh

    export SUMO_HOME=/the/path/to/sumo
    export PATH=/the/path/to/sumo/bin:$PATH

And source the :code:`.bashrc` file.

.. code-block:: sh

    source ~/.bashrc

Finally, you can test your SUMO installation is successful or not with the following commands:

.. code-block:: sh

    which sumo    # gives you the path
    sumo    # shows you the version of SUMO
    sumo-gui    # pop-up SUMO gui window

Install WOLF
~~~~~~~~~~~~

**Clone and setup conda env**
.. code-block:: sh

    git clone http://116.66.187.35:4502/gitlab/its/sow45_code.git
    cd sow45_code
    # create wolf env with all the dependencies
    conda env create -f environment.yml

**Install wolf with pip**

.. code-block:: sh

    cd sow45_code
    conda activate wolf
    pip install -e .

**or use wolf by specifying the PYTHONPATH**

Please add the following line into your :code:`.bashrc` file.

.. code-block:: sh

    export PYTHONPATH=$PYTHONPATH:/path/to/sow45_code

And source the :code:`.bashrc` file.

.. code-block:: sh

    source ~/.bashrc

At last, you might need to install optional packages, depending on your configuration:

.. code-block:: sh

    pip install tensorflow_gpu # for NVIDIA GPU
    pip install tensorflow-rocm # for AMD GPU
    pip install tensorflow-cpu # FOR CPU ONLY

And install torch:

.. code-block:: sh
    pip install torch # for NVIDIA or CPU only.





Installing flow
~~~~~~~~~~~~~~~

The original flow is a vehicle control oriented framework, which does not exactly match our tasks. But flow still has good and useful high-level simulation wrappers. Hence, we made lots of enhancements to let it fit our tasks. Here we'll introduce how to install the modified flow.

**TODO**: *Rephrase this paragraph*:
Before installing all the following packages, please make a blank virtual environment with any tool (conda/vertualenv/docker).

.. code-block:: sh

    # clone the source code from the github:
    git clone https://github.com/RaptorMai/flow.git
    # install the flow
    cd flow
    # dev branch
    git checkout sow5
    # switch to conda env wolf
    conda activate wolf
    # install flow on the conda wolf env
    pip install -e .






Installing QMIX Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing QMIX Dependencies: sacred, tensorboard-logger

.. code-block:: sh

    cd ..
    git clone https://github.com/oxwhirl/sacred.git
    cd sacred
    sed -i '36s/.*/        tf.random.set_seed(seed)/' randomness.py
    pip install -e .

    pip install tensorboard-logger
