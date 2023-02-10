FROM bentoml/model-server:0.11.0-py37
MAINTAINER ersilia

RUN pip install rdkit
RUN conda install pytorch==1.5.0 torchvision==0.6.0 -c pytorch
#RUN conda install pytorch==1.5.0 torchvision==0.6.0 cpuonly -c pytorch
RUN pip install rdkit
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install pandas	
RUN pip install matplotlib
RUN pip install ipython
RUN conda install openbabel=2.4.1 -c conda-forge
RUN pip install cairosvg

WORKDIR /repo
COPY . /repo
