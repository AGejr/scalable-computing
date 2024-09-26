# infrastructure

**Google cloud info**

[GPU availability by regions and zones](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones#view-using-table)

[GPU pricing](https://cloud.google.com/compute/gpus-pricing)

[GPU machine types](https://cloud.google.com/compute/docs/gpus)


## Prerequisites

**Tools:**
- gcloud
- terraform

**Google cloud cli authentication:**

Authenticate to Google cloud:
```
gcloud auth login
```

Initialize Google Cloud CLI:
```
gcloud init
```

Set up Application Default Credentials:
```
gcloud auth application-default login
```

## Getting started

1. **Create tfvars**

Create a `terraform.tfvars` file in this project to configure the project_id using the following example:

```
project_id = "your-project-id"
```

2. **Run terraform**

In `./backend` run:

```
terraform init
```

```
terraform apply -var-file=terraform.tfvars
```

3. **Authenticate kubeflow to cluster**

Follow this guide: [Google cloud Kubectl access](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl)

4. **Apply Kubeflow training operator**

Install Kubeflow Training operator v1.7.0:
```
kubectl apply -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.7.0"
```

5. **Apply tfjob**

```
kubectl apply -f ./jobs/mnist-example.yaml
```