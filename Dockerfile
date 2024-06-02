FROM python:3

WORKDIR /app

# RUN apt-get install build-essential -y
RUN apt-get upgrade -y
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5000

CMD ["python", "Api/app.py"]
