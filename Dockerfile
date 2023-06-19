# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt .
# Install pip requirements

# RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./deploy .

RUN python -m pip install -r requirements.txt

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["uvicorn", "main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"]
CMD ["uvicorn", "main:app", "--host=127.0.0.1", "--port=8000"]
