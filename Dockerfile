FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install flask pydantic openai

EXPOSE 7860

CMD ["python", "server/app.py"]