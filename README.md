# Background Remover Using U2Net
## Description
This repository is an easy example of background removal. I'm using [U2Net](https://github.com/xuebinqin/U-2-Net.git) model for creating musk and add some custom background. The main repo of this code is "https://github.com/axinc-ai/ailia-models-tflite".

## Instructions
### 1. clone the reprository
```cmd
git clone https://github.com/sazzad1779/Background_remover_using_u2net_.git
```

### 2. Create an environment
```cmd
python -m venv env_name
> env_name\scripts\activate   # for windows
> source env_name/bin/activate   # for ubuntu
```

### 3. To run this project
##### Install the pre-requisite libraries by running this command. 
```cmd
pip  install -r requirements.txt
```
##### To show blackwhite musk output image
```python 
python u2net.py -i samples
```

##### To save blackwhite musk output image
```python 
python u2net.py -i samples -s save_path
```
##### To show composite output image
```python 
python u2net.py -i samples --show
```
##### To show and save composite output image
```python 
python u2net.py -i samples  -s save_path -c  --show
```
# python u2net --help
```
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        give the model path
  -i INPUT, --input INPUT
                        give the input image
  -v VIDEO, --video VIDEO
                        give the video path
  -s SAVEPATH, --savepath SAVEPATH
                        path for the output (image / video).
  -b BACKGROUND_IMAGE, --background_image BACKGROUND_IMAGE
                        path for background image
  -c, --composite       Composite input image and predicted alpha value
  -w WIDTH, --width WIDTH
                        The segmentation width and height for u2net. (default: 320)
  --rgb                 Use rgb color space (default: bgr)
  --height HEIGHT       The segmentation height and height for u2net. (default: 320)
  --show                Show output image. 
```
# some result

<img align="right" src="https://github.com/sazzad1779/Background_remover_using_u2net_/blob/main/results/result_comp.jpg" width="100%" height="100%">
<img align="left" src="https://github.com/sazzad1779/Background_remover_using_u2net/blob/main/samples/input.png" width="45%" height="45%">
<img align="right" src="https://github.com/sazzad1779/Background_remover_using_u2net/blob/main/results/res_input.png" width="45%" height="45%">



 
