FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
COPY requirements.txt /tmp/

# setup the web proxy for Internet access

# configure the ubuntu's mirror
RUN apt-get update
RUN apt-get install -y wget git build-essential zip unzip


# install Miniconda (or Anaconda)
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh \
    && /bin/bash Miniconda3-4.5.4-Linux-x86_64.sh -b -p /softwares/miniconda3 \
    && rm -v Miniconda3-4.5.4-Linux-x86_64.sh
ENV PATH "/softwares/miniconda3/bin:${PATH}"
ENV LD_LIBRARY_PATH="/softwares/miniconda3/lib:${LD_LIBRARY_PATH}"

# install Python packages
RUN pip install --upgrade pip

# update conda
RUN conda update -n base conda
RUN conda config --add channels pytorch
RUN conda config --add channels rdkit
RUN conda config --add channels conda-forge
RUN conda config --add channels rmg
RUN conda install --yes --file /tmp/requirements.txt

# clean-up
RUN rm -rf /var/lib/apt/lists/*
RUN apt clean && apt autoremove -y

# provide defaults for the executing container
CMD [ "/bin/bash" ]
