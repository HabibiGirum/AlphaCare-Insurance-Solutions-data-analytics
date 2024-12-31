# Use the official Python image as the base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . /app

# Convert and execute the Jupyter Notebook
RUN jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis_executed.ipynb

# Default command
CMD ["cat", "analysis_executed.ipynb"]
