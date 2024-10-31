IMAGE_NAME=pytorch-ir-splitter
TAG=latest

CLUSTER_NAME ?= $(DEFAULT_CLUSTER_NAME)
CLUSTER_ZONE ?= $(DEFAULT_CLUSTER_ZONE)

build:
	docker build -t $(IMAGE_NAME):$(TAG) .

run:
	@if [ ! -d "./data/ir" ]; then \
		echo "Directory ./data/ir does not exist."; \
		exit 1; \
	fi
	docker run -v $(shell pwd)/data:/data $(IMAGE_NAME):$(TAG)

build-and-run: build run