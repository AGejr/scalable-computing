provider "google" {
  project = var.project_id
  region  = var.region
}

# Create GCS bucket
resource "google_storage_bucket" "ml_bucket" {
  name     = "ml-model-bucket-123456" # Must be globally unique
  location = "EU"
  force_destroy = true
  public_access_prevention = "enforced"

  lifecycle_rule {
    condition {
      age = 120
    }
    action {
      type = "Delete"
    }
  }
}

# GKE cluster
resource "google_container_cluster" "primary" {
  name               = "ml-gke-cluster"
  location           = var.region # Set to a European zone
  remove_default_node_pool = true
  initial_node_count = 1
  deletion_protection = false

  resource_labels = {
    environment = "dev"
  }

  ip_allocation_policy {}
}

# Node pool with GPU-enabled nodes
resource "google_container_node_pool" "gpu_nodes" {
  name       = "gpu-node-pool"
  cluster    = google_container_cluster.primary.name
  location   = google_container_cluster.primary.location

  node_config {
    machine_type = "n1-standard-4"
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    labels = {
      gpu = "true"
    }
    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
    }
    tags = ["gpu"]
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 5 # Adjust based on desired number of workers
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
