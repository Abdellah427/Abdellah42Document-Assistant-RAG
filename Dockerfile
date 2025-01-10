FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install --upgrade pip 

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]