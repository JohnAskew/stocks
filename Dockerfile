FROM python:3

RUN mkdir /opt/src 

WORKDIR /opt/src

ARG STOCK1=GOOG 
ARG STOCK2=MANH
ENV STOCK1=$STOCK1 STOCK2=$STOCK2

RUN set -ex; \
apt-get update \
&& apt-get install -y apt-utils ssmtp \
&& apt-get clean \ 
&& rm -rf /var/lib/apt/list/?*;  

COPY secret.key .

COPY Sent9.py .

COPY send_mail.py .

COPY ssmtp.conf .

RUN cat ssmtp.conf > /etc/ssmtp/ssmtp.conf; 

RUN echo "${STOCK1} ${STOCK2}"

CMD  python3 ./Sent9.py  "${STOCK1}" "${STOCK2}"
