# Installation
Installation suggestions for Ubuntu 16.04 LTS x86_64
For details see [https://github.com/facebookresearch/multipathnet](https://github.com/facebookresearch/multipathnet)
and [https://github.com/facebookresearch/deepmask](https://github.com/facebookresearch/deepmask)

## Update
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git
sudo apt-get install python-matplotlib
```

## Install NVidia driver
see [http://nvidia.com](http://nvidia.com)

## Install CUDA
```bash
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

## Install cuDNN
```bash
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod/8.0/cudnn-8.0-linux-x64-v5.1-tgz
tar xvf cudnn-8.0-linux-x64-v5.1-tgz
export LD_LIBRARY_PATH=~/cuda/lib64:$LD_LIBRARY_PATH

echo 'Add the following to your .bashrc:
export LD_LIBRARY_PATH=~/cuda/lib64:$LD_LIBRARY_PATH'
```

## Install Torch
```bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh

luarocks install cutorch
luarocks install tds
luarocks install inn
```

## Install COCO
```bash
git clone https://github.com/pdollar/coco.git ~/coco
cd ~/coco
luarocks make LuaAPI/rocks/coco-scm-1.rockspec
```

## Install OpenCV for data preparation
```bash
sudo apt-get install python-opencv
```

## Install Python and other dependencies
```bash
sudo apt-get install python-pip
sudo pip install numpy
sudo apt-get install libboost-all-dev
sudo apt-get install automake
sudo apt-get install cython

wget https://github.com/google/glog/archive/v0.3.3.zip
unzip v0.3.3.zip 
cd glog-0.3.3/
./configure
make
sudo make install

luarocks install inn
luarocks install torchnet
```

## MultiPathNet
```bash
git clone -b v1.0 https://github.com/facebook/thpp
cd thpp
git checkout d358a52  
cd thpp
THPP_NOFB=1 ./build.sh 

luarocks install fbpython
luarocks install class
luarocks install optnet

cd ~
cd coco/PythonAPI
make
export PYTHONPATH=$PYTHONPATH:~/coco/PythonAPI

echo 'Add the following to your .bashrc:
export PYTHONPATH=$PYTHONPATH:~/coco/PythonAPI'
```



























