IMAGE_NAME=pytorch-dist-mnist
TAG=latest
REGISTRY=gcr.io
DEFAULT_CLUSTER_NAME=autopilot-cluster
DEFAULT_CLUSTER_ZONE=us-central1

ifndef PROJECT_ID
$(error PROJECT_ID is not set. Please set it by running 'make <target> PROJECT_ID=<your_project_id>')
endif

CLUSTER_NAME ?= $(DEFAULT_CLUSTER_NAME)
CLUSTER_ZONE ?= $(DEFAULT_CLUSTER_ZONE)

upload-dataset:
	python3 -m venv venv
	. ./venv/bin/activate
	pip3 install -r requirements.txt
	python3 upload-dataset.py

build:
	docker build -t $(IMAGE_NAME):$(TAG) .

tag:
	docker tag $(IMAGE_NAME):$(TAG) $(REGISTRY)/$(PROJECT_ID)/$(IMAGE_NAME):$(TAG)

push:
	docker push $(REGISTRY)/$(PROJECT_ID)/$(IMAGE_NAME):$(TAG)

auth:
	gcloud auth configure-docker

build-and-push: build tag push

deploy:
	gcloud container clusters get-credentials $(CLUSTER_NAME) --zone $(CLUSTER_ZONE) --project $(PROJECT_ID)
	kubectl apply -f pytorch-mnist.yaml