FROM python:3.6-jessie

RUN apt update

#set working directory
WORKDIR ./app

#handle installing python dependencies
COPY ./requirements.txt /app/requirements.txt


#add folders and files to container
COPY ["./app.py", "./config.py", "./inference_pipeline.py", "./"]

#set environment variables
#server variables
ENV PORT 8080

RUN pip install -r /app/requirements.txt
RUN python inference_pipeline.py

# ENTRYPOINT ["/bin/bash"]

CMD ["gunicorn", "app:app", "--config=config.py"]

EXPOSE 8080
