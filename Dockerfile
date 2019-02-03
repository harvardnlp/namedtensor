FROM pytorch/pytorch:latest
RUN git clone https://github.com/harvardnlp/namedtensor.git && cd namedtensor && pip install -r requirements.txt && python setup.py install

