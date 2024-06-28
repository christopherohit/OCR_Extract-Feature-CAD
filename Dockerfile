FROM python:3.9

RUN apt-get update && apt upgrade -y && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /home/apprunner/

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the rest of the application code
COPY . .
RUN  pip install yolov10/.

# Expose the port
EXPOSE 8962

# Set the entrypoint
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8962"]

#   uvicorn main:app --host 0.0.0.0 --port 8962
