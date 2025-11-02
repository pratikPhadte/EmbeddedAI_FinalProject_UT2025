## This readme is for chapter 5 and 6

### Load the dataset - unzip cifar10 datset
### Open a terminal and go the cifar 10 directory
```
unzip <filename>
```

## Run inference on Kria CPU

### Open the tf-venv virtual environment
```
source tf-venv/bin/activate
```
### Call CPU inf script
```
python3 kria_inference_CPU.py --model cifar10_bw_model2.h5 --num_images 10
```
### Deactive the virtual env
```
deactivate
```

## Run inference on Kria DPU
Refer the DPU inference notebook