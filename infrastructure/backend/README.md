## Backend

### 1. Setup backend

The backend Terraform configuration creates a storage bucket in Google cloud to store the Terraform state.

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