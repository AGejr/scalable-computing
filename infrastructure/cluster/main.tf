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

  # Enable Autopilot mode
  enable_autopilot = true

  # Optional: Networking configuration
  network    = "default"
  subnetwork = "default"
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

resource "google_service_account" "gcs_access" {
  account_id   = "gcs-access-sa"
  display_name = "Service Account for GCS Access from GKE"
}

resource "google_storage_bucket_iam_member" "gcs_access" {
  bucket = google_storage_bucket.ml_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.gcs_access.email}"
}

resource "google_service_account_key" "gcs_access_key" {
  service_account_id = google_service_account.gcs_access.name
  public_key_type    = "TYPE_X509_PEM_FILE"
  private_key_type   = "TYPE_GOOGLE_CREDENTIALS_FILE"
}

resource "kubernetes_secret" "gcs_access" {
  metadata {
    name = "gcs-access"
    namespace = "default"
  }

  data = {
    "service-account.json" = base64encode(google_service_account_key.gcs_access_key.private_key)
  }
}