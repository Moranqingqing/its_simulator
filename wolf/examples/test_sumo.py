import os, sys

if not 'SUMO_HOME' in os.environ:
    sys.exit("please declare environment variable 'SUMO_HOME', or set it using os.environ e.g. os.environ['SUMO_HOME'] =  \"/home/user/sumo_binaries/bin\"")
