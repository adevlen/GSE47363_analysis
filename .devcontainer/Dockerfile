# base image (with Conda)
FROM continuumio/miniconda3:latest

# must be root to install system packages
USER root

# add env so that user input won't be asked for during install
ENV DEBIAN_FRONTEND=noninteractive

# set working directory
WORKDIR /work

# install system-level compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# copy requirements file into working directory
COPY environment.yml ./

# build environment
RUN conda env create -f environment.yml

# force environment into system PATH
ENV PATH="/opt/conda/envs/anna_analysis/bin:${PATH}"

# bash setup
RUN conda init bash
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate anna_analysis" >> ~/.bashrc

# activate environment by default
SHELL ["/bin/bash", "--login", "-c"]