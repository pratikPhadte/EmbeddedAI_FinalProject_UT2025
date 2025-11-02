## This readme is for chapter 4 

### Clone DPU-PYNQ repo
```
git clone https://github.com/Xilinx/DPU-PYNQ.git
```

### Source vivado 2022.2 and XRT
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
