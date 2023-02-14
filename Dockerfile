FROM bentoml/model-server:0.11.0-py37
MAINTAINER ersilia

RUN pip install rdkit==2022.9.4
RUN pip install scipy==1.7.3
RUN pip install scikit-learn==1.0.2
RUN pip install pandas==1.3.5
RUN pip install matplotlib==3.5.3
#RUN pip install ipython
RUN conda install openbabel=2.4.1 -c conda-forge
RUN conda install pytorch==1.5.0 torchvision==0.6.0 cpuonly -c pytorch
#RUN pip install cairosvg

WORKDIR /repo
COPY . /repo
