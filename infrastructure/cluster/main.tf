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

resource "google_container_cluster" "autopilot_cluster" {
  name     = "autopilot-cluster"
  location = var.region

  # Enable Autopilot mode
  enable_autopilot = true

  # Optional: Networking configuration
  network    = "default"
  subnetwork = "default"
}
