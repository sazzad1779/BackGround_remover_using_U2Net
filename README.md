# Background Remover Using U2Net
## Description
This repository is an easy example of background removal. I'm using U2Net model for creating musk and add some custom background. The main repo of this code is "https://github.com/axinc-ai/ailia-models-tflite".

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
![no image found](https://www.google.com/url?sa=i&url=https%3A%2F%2Fcolors.artyclick.com%2Fcolor-names-dictionary%2Fcolor-names%2Fblood-red-color&psig=AOvVaw3LkyH0HYJNKW0SG8j2-x_V&ust=1696231095698000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCJj7ta-n1IEDFQAAAAAdAAAAABAJ)
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



 
