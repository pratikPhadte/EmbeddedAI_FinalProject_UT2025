# This readme is for chapter 4.5, 5 and 6

# Chapter 4.5
# Setting Up Internet Access and TensorFlow Installation on Kria

## 1. Verify Internet Access
```bash
ping -c 3 proxy.utwente.nl
```

---

## 2. Configure Proxy Settings
Export proxy variables for both HTTP and HTTPS connections:
```bash
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"
export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export no_proxy="localhost,127.0.0.1"
```

Then, reload the environment:
```bash
source /etc/environment
```

---

## 3. Configure APT Proxy
Edit the APT proxy configuration file:
```bash
sudo nano /etc/apt/apt.conf.d/95proxies
```

Add these lines below
```bash
Acquire::http::Proxy "http://proxy.utwente.nl:3128";
Acquire::https::Proxy "http://proxy.utwente.nl:3128";
Acquire::ftp::Proxy "http://proxy.utwente.nl:3128";
```
Save and Exit nano


## 4. Install TensorFlow with Proxy Support

### Create a Python Virtual Environment
```bash
python3 -m venv tf-venv
```

### Activate the Virtual Environment
```bash
source tf-venv/bin/activate
```

### Install TensorFlow via Proxy
```bash
pip install tensorflow==2.11.0 --proxy="http://proxy.utwente.nl:3128"
```

---

## 5. Fix NumPy Compatibility
Uninstall any incompatible NumPy versions:
```bash
pip uninstall numpy
```

Then, install a compatible version:
```bash
pip install "numpy<2.0" --proxy="http://proxy.utwente.nl:3128"
```

---

## 6. Verify TensorFlow Installation
Run Python and check if TensorFlow imports successfully:
```bash
python3
```

Then inside Python:
```python
import tensorflow as tf
import numpy as np
print(tf.__version__)
print(np.__version__)
```

If it prints `2.11.0`, your installation is successful! 

---


# Chapter 5
### Load the dataset - download the cifar-10 dataset on your Local machine and then Upload to Kria
```
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```
### Unzip cifar10 datset
### Open a terminal and go the cifar 10 directory
```bash
unzip <filename>
```

# Chapter 6
# Run inference on Kria CPU

### Open the tf-venv virtual environment
```bash
source tf-venv/bin/activate
```
### Call CPU inf script
```python  
python3 kria_inference_CPU.py --model cifar10_bw_model2.h5 --num_images 10
```
### Deactive the virtual env
```bash
deactivate
```

## Run inference on Kria DPU
Refer the DPU inference notebook