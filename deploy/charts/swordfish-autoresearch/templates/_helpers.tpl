{{/* Common labels — same shape as ai-training-infra so kubectl filters work. */}}
{{- define "swordfish-autoresearch.labels" -}}
app.kubernetes.io/name: swordfish-autoresearch
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: swordfish
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app: swordfish-autoresearch
{{- end }}

{{/* Job name with a per-release suffix so re-installs don't collide. */}}
{{- define "swordfish-autoresearch.jobName" -}}
swordfish-profile-{{ .Release.Name | trunc 30 | trimSuffix "-" }}
{{- end }}
