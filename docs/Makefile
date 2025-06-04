.PHONY: build serve clean docker-build docker-serve

# Build the documentation site
build:
	mkdocs build

# Serve the documentation site locally
serve:
	mkdocs serve

# Clean the build directory
clean:
	rm -rf site/

# Build the documentation Docker image
docker-build:
	docker-compose build

# Serve the documentation using Docker
docker-serve:
	docker-compose up

# Stop and remove Docker containers
docker-down:
	docker-compose down

# Build and deploy documentation
deploy: build
	mkdocs gh-deploy --force 