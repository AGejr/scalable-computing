variable "project_id" {
  description = "Your Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "The Google Cloud region for the bucket"
  type        = string
  default     = "EU"
}

variable "gcs_bucket_name" {
  description = "A globally unique name for your GCS bucket"
  type        = string
  default     = "terraform-state-bucket-123456"
}
