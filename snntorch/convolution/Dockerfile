FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y git wget vim ghdl gtkwave && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install snntorch tonic matplotlib tqdm brevitas vunit_hdl seaborn

WORKDIR /workspace
CMD ["bash"]
