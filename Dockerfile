ARG OSNICK=bionic
ARG TARGET=cpu

FROM redislabs/redisai:edge-${TARGET}-${OSNICK}

RUN apt update && apt install -y python3 python3-pip
ADD . /build
WORKDIR /build
RUN pip3 install -r requirements.txt
RUN poetry config virtualenvs.create false
RUN poetry install
RUN poetry build
RUN pip3 install dist/redisai*.tar.gz
