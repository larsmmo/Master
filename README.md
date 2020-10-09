# Master thesis - Autumn 2020

Code related to master thesis by Lars Martin Moe (larsmmo), for the Cybernetics and Robotics study at NTNU, Trondheim.

## Setup

Setup assumes you have [Anaconda](https://www.anaconda.com/distribution/#download-section) (Python 3.6 or greater) on your system. If you do not, please download and install Anaconda before proceding.

The next step is to clone or download the tutorial materials in this repository. If you are familiar with Git, run the clone command:

    git clone https://github.com/larsmmo/oxygenModel
    
or download a zipfile and extract its contents on your computer.
***
The repository contains a file called `environment.yml` that includes a list of all required packages. If you run:

    conda env create
    
from the main directory, the environment will be created and all listed packages installed. Enable the environment using:

    conda activate oxygenModel
    
**JupyterLab** can be started by typing:

    jupyter lab