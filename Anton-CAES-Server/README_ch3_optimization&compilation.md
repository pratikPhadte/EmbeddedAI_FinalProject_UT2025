
## This readme is for chapter 3
### This folder contains the Vitis-ai-quantization script that does Post training quantization on the .h5 model

## STEPS

### Upload the .h5 model (Generated on Utwente Jupyter Lab) on the Anton server
```bash
scp /path/to/local/directory your_username@anton.ewi.utwente.nl:/path/to/remote/destination

```
### Load the dataset - download the cifar-10 dataset
```
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```
### Unzip cifar10 datset
```bash
unzip <filename>
```

### Access Vitis-AI and tensorflow 2 on Anton Caes server

```bash
apptainer run /remote/labware/containers/vitis-ai_2.5.sif
```
```bash
conda activate vitis-ai-tensorflow2
```

### Do post training quantization on your model
```bash
(vitis-ai-tensorflow2) Apptainer> python vitis-ai-quantization_grayscale.py 
```

### Compile the model for DPU target
```bash
vai_c_tensorflow2 -m "<path_to_your_quantized_model>" -a "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json" -o "<path_to_store_your_compiled_model>" -n <compiled_model_name>

```