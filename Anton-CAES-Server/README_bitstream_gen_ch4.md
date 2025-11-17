## This readme is for chapter 4 

### The Kria has an FPGA on it, to program it, you need to add the overlay of a DEEP PROCESSING UNIT (DPU), below steps are to generate the bitstream for the DPU and then upload it to the Kria's FPGA.

### In this step, you need to ask yourself how you can make the DPU specific optimization such that it is able to process the model layers, ops better. You may check if majority of your model layers that are supported by DPU. You can also check what type of data does the DPU expect. 

### Clone DPU-PYNQ repo for 3.0.1 version
```
https://github.com/Xilinx/DPU-PYNQ.git
```

### Source vivado 2022.2 and XRT in your server's terminal
```bash
module load xilinx/vivado/2022.2
module load xilinx/vitis/2022.2
module load xilinx/xrt
```

go in root and type below command in terminal to source XRT
```bash
source opt/xilinx/xrt/setup.sh
```

### Go to DPU-PYNQ repo & generate the bitstream after configuring the DPU as per your need
```bash
make BOARD=kv260_som
```


## Additional: If you want to see the DPU design, DPU bus connections, Slices used and see the generation reports, you can open vivado to inspect.
```bash
module load xilinx/vivado/2022.2
vivado
```
