FROM python:3.6-jessie

RUN apt update

WORKDIR /app

ADD requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

ADD . /app

ADD inference_pipeline.py /app/inference_pipeline.py

ENV PORT 8080

RUN python inference_pipeline.py

RUN ls

CMD ["gunicorn", "app:app", "--config=config.py"]