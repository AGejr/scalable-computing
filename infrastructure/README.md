# infrastructure

**Prerequisite tools:**
- gcloud
- terraform (v1.9.6)

## Getting started

### Setup backend

The backend Terraform configuration creates a storage bucket in Google cloud to store the Terraform state.

1. **Authenticate gcloud**

Initialize Google Cloud CLI:
```
gcloud init
```

Authenticate to Google cloud using Google Account:
```
gcloud auth application-default login
```

2. **Create tfvars**

Create a `terraform.tfvars` file in this project to configure the project_id using the following example:

```
project_id = "your-project-id"
```

3. **Run terraform**

In `./backend` run:

```
terraform init
```

```
terraform apply -var-file=terraform.tfvars
```

### Create cluster

Install NVIDIA drivers:
```
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/daemonset.yaml
```

Install Kubeflow Training operator v1.7.0:
```
kubectl apply -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.7.0"
```

[Google cloud Kubectl access](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl)

[Scaling issue](https://cloud.google.com/kubernetes-engine/docs/troubleshooting/autopilot-clusters#scale-up-failed-serial-port-logging):
```
gcloud compute project-info add-metadata --metadata serial-port-logging-enable=true
```

[GPU availability by regions and zones](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones#view-using-table)

[GPU pricing](https://cloud.google.com/compute/gpus-pricing)

[GPU machine types](https://cloud.google.com/compute/docs/gpus)

