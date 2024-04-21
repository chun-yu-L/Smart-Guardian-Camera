# Specify the base image (Ultralytics)
FROM ultralytics/ultralytics

# Copy requirements.txt file to the container
COPY ./ /app/

# Chage working directory
WORKDIR /app

# Install required python modules listed in requirements.txt
RUN pip install -r requirements.txt

# Expose port
ENV APP_PORT=8888

# Run FastAPI app. Default port is 8888 and can be overridden with --port
CMD uvicorn main:app --host 0.0.0.0 --port $APP_PORT
