provider "google" {
  project = var.project_id
  region  = var.region
}

provider "kubernetes" {
  host                   = google_container_cluster.autopilot_cluster.endpoint
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.autopilot_cluster.master_auth.0.cluster_ca_certificate)
}

data "google_client_config" "default" {}

resource "google_container_cluster" "autopilot_cluster" {
  name     = "autopilot-cluster"
  location = var.region

  deletion_protection = false

  # Enable Autopilot mode
  enable_autopilot = true

  # Optional: Networking configuration
  network    = "default"
  subnetwork = "default"

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  # Enable GcsFuseCsiDriver addon
  addons_config {
    gcs_fuse_csi_driver_config {
      enabled = true
    }
  }
}

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

# Create GCP Service Account
resource "google_service_account" "gcs_access" {
  account_id   = "gcs-access-sa"
  display_name = "Service Account for GCS Access from GKE"
}

# Grant access to GCS bucket
resource "google_storage_bucket_iam_member" "gcs_access" {
  bucket = google_storage_bucket.ml_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.gcs_access.email}"
}

# Kubernetes Service Account
resource "kubernetes_service_account" "ksa_gcs_access" {
  metadata {
    name        = "ksa-gcs-access"
    namespace   = "default"
    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.gcs_access.email
    }
  }
}

# IAM Policy Binding
resource "google_service_account_iam_binding" "binding" {
  service_account_id = google_service_account.gcs_access.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[default/ksa-gcs-access]"
  ]
}
