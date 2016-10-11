# Preparation

## CUDA

Install CUDA Toolkit 8.0 at default location (/Developer/NVIDIA/CUDA-8.0), Xcode 7.3, (optional CL tools) on OS X 10.11.

Use xcode-select to set active Xcode to 7.3.

CUDA 8.0 Toolkit only supports Xcode 7.3-.

You'll need an Nvidia graphics card to run CUDA.

## Nvidia OptiX

Obtain Nvidia OptiX [here](https://developer.nvidia.com/optix) and install it at default location.

## Cinder

Get cinder [here](https://github.com/cinder/cinder), build, and set the "CINDER_PATH" macro in build settings to point to your cinder root.

# Existing Cinder Xcode project configuration notes

If you would like to include OptiX in an exisiting project, follow these steps:

In Build Settings tab of your target,

* add to Library Search Paths:

	`/Developer/OptiX/lib64`

* add to Header Search Paths:
	
	`/Developer/OptiX/include`
	`/usr/local/cuda/include`

* set Runpath Search Paths to:

	`/Developer/OptiX/lib64`


In Build Phases,

* add to Link Binary With Libraries:

	`/Developer/OptiX/lib64/liboptix.1.dylib`


In Build Rules,

* create a custom build rule for all *.cu files:

	```
	mkdir -p ${PROJECT_DIR}/../assets/ptx
	
	/usr/local/cuda/bin/nvcc -ptx -o ${PROJECT_DIR}/../assets/ptx/${INPUT_FILE_BASE}.cu.ptx -I../include -I/Developer/OptiX/include ${INPUT_FILE_PATH}
	```

	and set output files to:

	`${PROJECT_DIR}/../assets/ptx/${INPUT_FILE_BASE}.cu.ptx`
