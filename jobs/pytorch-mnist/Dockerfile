FROM python:3.12.0

RUN pip install tensorboardX==2.6.2 \
    torch==2.4.1 \
    torchvision==0.19.1 \
    matplotlib \
    scikit-learn
RUN mkdir -p /opt/mnist

WORKDIR /opt/mnist/src
ADD mnist.py /opt/mnist/src/mnist.py

RUN chgrp -R 0 /opt/mnist \
    && chmod -R g+rwX /opt/mnist

ENTRYPOINT ["python", "/opt/mnist/src/mnist.py"]