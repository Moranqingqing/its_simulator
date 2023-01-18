import os

os.system("rm -rf build")
os.system("make html")
os.system("scp -r build ncarrara@137.74.194.42:/home/ncarrara/website/public/others")