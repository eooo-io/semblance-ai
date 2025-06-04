FROM squidfunk/mkdocs-material:latest

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Set working directory
WORKDIR /docs

# Copy documentation source
COPY . .

# Expose MkDocs development server port
EXPOSE 8000

# Start development server by default
CMD ["mkdocs", "serve", "--dev-addr=0.0.0.0:8000"] 