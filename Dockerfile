FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /workspace/AICB

COPY . /workspace/AICB

RUN mv ./workload_generator /usr/local/lib/python3.10/dist-packages &&\
    mv ./utils /usr/local/lib/python3.10/dist-packages &&\
    mv ./log_analyzer /usr/local/lib/python3.10/dist-packages &&\
    pip install git+https://github.com/fanshiqing/grouped_gemm@v1.0 &&\
    pip3 install einops  && \
    echo 'install deep_gemm for DeepSeek AIOB' && \
    pip3 install git+https://github.com/deepseek-ai/DeepGEMM@391755ada0ffefa9a6a52b6f14dcaf22d1a463e0

