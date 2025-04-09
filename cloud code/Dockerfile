FROM python:3.9

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
COPY service-account.json /app/service-account.json

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account.json"

EXPOSE 8080

CMD ["python", "app.py"]
