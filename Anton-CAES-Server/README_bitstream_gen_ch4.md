## This readme is for chapter 4 

### In this step, you need to ask yourself how you can make the DPU specific optimization such that it is able to process the model layers, ops better. You may check if majority of your model layers that are supported by DPU. You can also check what type of data does the DPU expect. 

### Clone DPU-PYNQ repo for 3.0.1 version
```
https://github.com/Xilinx/DPU-PYNQ.git
```

### Source vivado 2022.2 and XRT in your server's terminal
```
module load xilinx/vivado/2022.2
module load xilinx/xrt
```
```
#in root
source opt/xilinx/xrt/setup.sh
```

### Go to DPU-PYNQ repo & generate the bitstream after configuring the DPU as per your need
```
make BOARD=kv260_som
```
