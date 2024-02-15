# Specify the base image (Ultralytics)
FROM ultralytics/ultralytics

# Copy requirements.txt file to the container
COPY requirements.txt /requirements.txt

# Chage working directory
WORKDIR /app

# Install required python modules listed in requirements.txt
RUN pip install -r /requirements.txt
