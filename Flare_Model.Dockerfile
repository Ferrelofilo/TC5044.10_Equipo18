# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /TC5044.10_Equipo18

# Copy the current directory contents into the container at /app
COPY . /TC5044.10_Equipo18

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install dvc[gdrive]

# Make port 80 available to the world outside this container
EXPOSE 8000
EXPOSE 5000

# Run app.py when the container launches
# CMD ["python", "app.py"]