FROM bentoml/model-server:0.11.0-py38
MAINTAINER ersilia

RUN pip install rdkit==2022.9.5
RUN pip install scipy==1.7.3
RUN pip install scikit-learn==1.0.2
RUN pip install pandas==1.3.5
RUN pip install matplotlib==3.5.3
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
RUN conda install -c conda-forge openbabel=3.1.1 libarchive=3.7.7

WORKDIR /repo
COPY . /repo
