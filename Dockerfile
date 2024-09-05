FROM python:3.12

WORKDIR /app

COPY main.py /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]

EXPOSE 8888
