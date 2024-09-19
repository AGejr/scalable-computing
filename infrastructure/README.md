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

...