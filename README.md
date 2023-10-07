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
# some result
<img align="left" src="https://github.com/sazzad1779/Background_remover_using_u2net/blob/main/samples/input.png" width="45%" height="45%">
<img align="right" src="https://github.com/sazzad1779/Background_remover_using_u2net/blob/main/results/res_input.png" width="45%" height="45%">




 
