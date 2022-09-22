# EduOptimNeurostim
GP-BO optimization of neurostimulation. 
Educational material.

Companion code for the following paper : [Link to paper available upon publication]

Please cite the present code as:
Bonizzato M.<sup>&</sup>, Macar U.<sup>&</sup>, Guay-Hottin R., Choinière L., Lajoie G., Dancause N. 2021. “EduOptimNeurostim”, GitHub.  https://github.com/mbonizzato/EduOptimNeurostim/. \
<sup>&</sup>, these authors equally contributed to the repository concept and material.


## Usage notes

The library is built around main.py and a JSON file containing all the necessary parameters ([config](/config) folder). 
We built a series of tutorials to help navigate all the available features and design library calls ([tutorials](/tutorials) folder).
A few utilities for data downloading and result visualization complement this release ([scripts](/scripts) folder).

You may want to run our Tutorial 1 first.

### Getting started

1. Make sure you are equipped to read an IPYNB file, a notebook document created by Jupyter Notebook.
https://www.anaconda.com/products/distribution

2. The notebooks contain !bash commands to download the dataset. If you wish to use these, you will need a git bash.
https://gitforwindows.org/

3. Clone the repository.

``` git clone https://github.com/mbonizzato/EduOptimNeurostim.git ```

4. You will need python 3.7.4 to run the code. We suggest to create a virtual environment:

```
python3.7.4 -m venv <virtual-environment-name>
source <virtual-environment-name>/bin/activate
```

5. Install the requirements:

  ``` pip install -r requirements.txt ```

6. Open the notebook Experiment 1 in Jupyter Notebook and enjoy the tutorial!

## Data

The dataset can be downloaded at :  https://osf.io/54vhx/
While running the provided tutorial notebooks, the dataset will be automatically downloaded for you.

Extensive dataset explanation is available at the project's Wiki : https://osf.io/54vhx/wiki/home/

Please cite the dataset as:
Bonizzato M., Massai E.<sup>&</sup>, Côté S.<sup>&</sup>, Quessy S., Martinez M., and Dancause N. 2021. “OptimizeNeurostim” OSF. osf.io/54vhx. \
<sup>&</sup>, these authors equally contributed to the dataset.






