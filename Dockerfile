FROM tiangolo/uwsgi-nginx-flask:python3.6
#FROM python:3.6.1-alpine

WORKDIR /app/

COPY requirements.txt /app/
#RUN pip install --proxy http://user:password@proxy.in.iantel.com.uy:80 -r ./requirements.txt
RUN pip install -r ./requirements.txt

ENV ENVIRONMENT testing

COPY ./ApiREST/fiveClassSVC_model.sav /app/
COPY ./ApiREST/Tfidf_vect.pkl /app/
COPY ./ApiREST/TablaPerformance.csv /app/

COPY ./ApiREST/HmNLP.py /app/
COPY ./ApiREST/clasifica.py /app/

CMD ["python","/app/clasifica.py"]