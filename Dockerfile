ARG OSNICK=bionic
ARG TARGET=cpu

FROM redislabs/redisai:edge-${TARGET}-${OSNICK} as builder

RUN apt update && apt install -y python3 python3-pip
ADD . /build
WORKDIR /build
RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry build

### clean docker stage
FROM redislabs/redisai:edge-${TARGET}-${OSNICK} as runner

RUN apt update && apt install -y python3 python3-pip
RUN rm -rf /var/cache/apt/

COPY --from=builder /build/dist/redisai*.tar.gz /tmp/
RUN pip3 install /tmp/redisai*.tar.gz
