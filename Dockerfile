FROM python:3.10-slim

WORKDIR /code

# Copy requirements and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Download the specific spacy model required by your app.py
RUN python -m spacy download en_core_web_sm

# Copy everything else over
COPY . .

# Expose the mandatory Hugging Face traffic port
EXPOSE 7860

# Command to execute Flask app
CMD ["python", "app.py"]