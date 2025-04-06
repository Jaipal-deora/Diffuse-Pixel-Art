# Diffuse Pixel Art 
Diffuse Pixel Art uses diffusion architecture to generate pixel art models. 

# Create set up 
## Create virtual environment 
``` conda create -n diffuse_pixel python=3.10 -y ``` 

## Activate virtual env 
``` conda activate diffuse_pixel```

## Install required packages 
``` pip install -r requirements.txt```

## run the streamlit app 
```streamlit run app.py```

# docker instructions 
``` docker build -t diffuse_pixel_st .```