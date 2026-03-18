# base image (with Conda)
FROM continuumio/miniconda3:latest

# add env so that user input won't be asked for during install
ENV DEBIAN_FRONTEND=noninteractive

# set working directory
WORKDIR /work

# copy requirements file into working directory
COPY environment.yml ./

# build environment
RUN conda env create -f environment.yml

RUN conda init bash

# activate environment by default
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate anna_analysis" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]