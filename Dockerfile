FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /workspace/AICB

Copy . /workspace/AICB

RUN mv ./workload_generator /usr/local/lib/python3.10/dist-packages &&\
    mv ./utils /usr/local/lib/python3.10/dist-packages &&\
    mv ./log_analyzer /usr/local/lib/python3.10/dist-packages &&\
    pip3 install einops 


