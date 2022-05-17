FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04


# FROM nvcr.io/nvidia/pytorch:21.x02-py3
# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/home/user/miniconda/bin:$PATH

COPY environment.yml /app/environment.yml
COPY / /app/

#download miniconda and setup
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda update -n base -c defaults conda \
 && conda env update -n base -f /app/environment.yml \
 && rm /app/environment.yml \
 && conda clean -ya



RUN pwd
RUN ls

# Set the default command to python3
CMD ["python3"]

#run python files
CMD ["python3", "tests/test.py"]
CMD ["python3", "tests/collect_env.py"]