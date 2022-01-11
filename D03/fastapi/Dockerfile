# the base image
FROM python:3.7.3

# main working dir for subsequent commands
WORKDIR /app

# copy the file containing the necessary python libraries and install them to the image
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# copy the saved model and the web apps
COPY src /app/src
COPY data /app/data

# run the server
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]