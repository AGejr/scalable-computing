output "gcs_bucket_name" {
  description = "The name of the GCS bucket created"
  value       = google_storage_bucket.terraform_state.name
}
