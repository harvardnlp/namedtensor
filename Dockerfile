FROM pytorch/pytorch:latest
RUN git clone https://github.com/harvardnlp/NamedTensor.git && cd NamedTensor && pip install -r requirements.txt && python setup.py install

