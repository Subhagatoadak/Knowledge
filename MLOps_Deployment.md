# 🚀 ML/AI Deployment Tooling Across AWS, Azure, and GCP (Production-Grade)

> **Status**: Work in Progress 🚧

This document provides a comprehensive list of services and tools used to deploy production-grade machine learning and AI applications across the three major cloud platforms: **AWS**, **Azure**, and **GCP**. The tools are categorized by the deployment lifecycle stages.

---

## 🔄 ML Deployment Lifecycle Stages

1. Data Ingestion & Storage
2. Data Processing & Feature Engineering
3. Model Training
4. Experiment Tracking
5. Model Registry
6. Model Deployment & Serving
7. Monitoring & Retraining
8. CI/CD & DevOps Integration
9. Security & IAM
10. Load Balancing & Auto Scaling
11. Distributed Training & Resource Optimization
12. Edge & On-device Deployment
13. Feature Store & Model Packaging
14. Testing, A/B Validation & Shadow Deployment
15. Data Cataloging & Lineage
16. MLOps & Orchestration

---

## 🟨 AWS (Amazon Web Services)

| Stage                | Tool/Service                                                 | Description                                |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| Data Ingestion       | Kinesis, Glue, Data Pipeline                                 | Stream or batch data ingestion             |
| Data Storage         | S3, RDS, Redshift, DynamoDB                                  | Object, relational, and NoSQL storage      |
| Data Processing      | Glue, EMR, Lambda                                            | ETL, Spark, and event-based compute        |
| Model Training       | SageMaker, EC2, ECS, Horovod, SageMaker MPI                  | Managed or distributed training            |
| Experiment Tracking  | SageMaker Experiments, CloudWatch, MLflow                    | Manage runs and metrics                    |
| Model Registry       | SageMaker Model Registry                                     | Track versions and metadata                |
| Model Deployment     | SageMaker Endpoints, Lambda, ECS/EKS                         | Realtime/batch serving                     |
| Load Balancing       | Elastic Load Balancer (ELB), Application Load Balancer (ALB) | Distribute traffic and ensure availability |
| Monitoring           | CloudWatch, SageMaker Monitor, X-Ray, Prometheus             | Monitor drift, latency, metrics            |
| CI/CD                | CodePipeline, CodeBuild, GitHub Actions                      | Automate ML workflows                      |
| Security             | IAM, KMS, VPC, SageMaker Roles                               | Access and encryption management           |
| Edge Deployment      | AWS Greengrass, SageMaker Neo, IoT Core                      | Deploy on devices or edge nodes            |
| Model Packaging      | Docker, Conda, TorchScript, ONNX                             | Model serialization and containerization   |
| Data Cataloging      | Glue Data Catalog, Lake Formation                            | Schema, lineage, and catalog management    |
| Testing & Validation | A/B testing with SageMaker Shadow Deployment, Canary Testing | Model testing in production                |
| MLOps                | SageMaker Pipelines, Step Functions                          | End-to-end pipeline orchestration          |

---

## 🟦 Azure (Microsoft Azure)

| Stage                | Tool/Service                                          | Description                            |
| -------------------- | ----------------------------------------------------- | -------------------------------------- |
| Data Ingestion       | Data Factory, Event Hubs, IoT Hub                     | ETL and real-time feeds                |
| Data Storage         | Blob Storage, SQL DB, Cosmos DB, Data Lake            | Multiple storage options               |
| Data Processing      | Databricks, Synapse, Functions                        | Notebooks, Spark, serverless ETL       |
| Model Training       | Azure ML, Databricks, PyTorch DDP                     | Notebooks, distributed training        |
| Experiment Tracking  | Azure ML Experiments, App Insights, MLflow            | Experiment lifecycle management        |
| Model Registry       | Azure ML Model Registry                               | Manage and deploy trained models       |
| Model Deployment     | Azure ML Endpoints, AKS, Functions                    | Scalable realtime or batch inference   |
| Load Balancing       | Azure Load Balancer, Azure Application Gateway        | Load distribution and failover control |
| Monitoring           | Azure Monitor, ML Monitor, Prometheus + Log Analytics | Track metrics, bias, and drift         |
| CI/CD                | Azure DevOps, GitHub Actions                          | Automate model workflows               |
| Security             | RBAC, Key Vault, VNET                                 | Role and secret management             |
| Edge Deployment      | Azure IoT Edge, Azure Percept                         | Run models on edge devices             |
| Model Packaging      | Docker, Conda, ONNX, Azure ML Environments            | Serialization and portability          |
| Data Cataloging      | Azure Purview                                         | Metadata, governance, and lineage      |
| Testing & Validation | A/B testing, blue-green deployment in AKS             | Controlled rollout of models           |
| MLOps                | Azure ML Pipelines, Logic Apps, Data Factory          | Orchestrated ML pipelines              |

---

## 🟥 GCP (Google Cloud Platform)

| Stage                | Tool/Service                                                       | Description                            |
| -------------------- | ------------------------------------------------------------------ | -------------------------------------- |
| Data Ingestion       | Pub/Sub, Dataflow, Transfer Service                                | Streaming or batch ingestion           |
| Data Storage         | GCS, BigQuery, Firestore, Cloud SQL                                | All data storage modalities            |
| Data Processing      | Dataflow, Dataproc, Functions                                      | Batch, stream, or function-based ETL   |
| Model Training       | Vertex AI, AI Platform, GKE, TPUs                                  | Scalable training workloads            |
| Experiment Tracking  | Vertex AI Experiments, TensorBoard, MLflow                         | Metrics and comparison                 |
| Model Registry       | Vertex AI Model Registry                                           | Track and serve trained models         |
| Model Deployment     | Vertex AI Endpoints, Cloud Run, GKE                                | Serve models via APIs                  |
| Load Balancing       | Cloud Load Balancing (HTTP/HTTPS/Network), Internal Load Balancing | Distribute model inference loads       |
| Monitoring           | Vertex AI Monitoring, Stackdriver, Prometheus                      | Drift, skew, and logs monitoring       |
| CI/CD                | Cloud Build, GitHub Actions, Tekton                                | Full CI/CD pipelines                   |
| Security             | IAM, Secret Manager, VPC SC                                        | Policies, secrets, network controls    |
| Edge Deployment      | Coral TPU, Edge TPU Compiler, IoT Core                             | Run inference on device edge           |
| Model Packaging      | Docker, ONNX, TensorFlow Lite                                      | Format conversion and containerization |
| Data Cataloging      | Data Catalog, BigLake Metadata                                     | Metadata, governance, data lineage     |
| Testing & Validation | Shadow testing, staged rollout, rollout percentages                | Safe model experimentation             |
| MLOps                | Vertex AI Pipelines, Cloud Composer                                | Managed orchestration and DAGs         |

---

## 🔁 Cloud Platform Tool Equivalents

| Functionality            | AWS                         | Azure                             | GCP                                  |
| ------------------------ | --------------------------- | --------------------------------- | ------------------------------------ |
| Managed ML Platform      | SageMaker                   | Azure ML                          | Vertex AI                            |
| Model Registry           | SageMaker Registry          | Azure ML Registry                 | Vertex AI Registry                   |
| Serving Models           | SageMaker Endpoints         | Azure ML Endpoints / AKS          | Vertex AI Endpoints / Cloud Run      |
| Pipelines                | SageMaker Pipelines         | Azure ML Pipelines                | Vertex AI Pipelines                  |
| Monitoring               | SageMaker Monitor           | Azure Monitor                     | Vertex AI Monitoring                 |
| Feature Store            | SageMaker Feature Store     | Azure ML Feature Store            | Vertex AI Feature Store              |
| Orchestration            | Step Functions              | Data Factory / Logic Apps         | Cloud Composer                       |
| CI/CD                    | CodePipeline + CodeBuild    | Azure DevOps                      | Cloud Build / Tekton                 |
| Notebook IDE             | SageMaker Studio            | Azure ML Notebooks                | Vertex AI Workbench                  |
| Load Balancer            | ELB / ALB                   | Azure Load Balancer / App Gateway | Cloud Load Balancing                 |
| Auto Scaling             | ASG / EKS HPA               | AKS Auto-scaler / VMSS            | GKE Autopilot / Cloud Run Autoscaler |
| Distributed Training     | Horovod / MPI               | PyTorch DDP / Databricks          | TPUs / Distributed Trainer API       |
| Edge Deployment          | Greengrass / Neo            | IoT Edge / Percept                | Coral TPU / Edge TPU Compiler        |
| Data Cataloging          | Glue / Lake Formation       | Azure Purview                     | GCP Data Catalog                     |
| Model Packaging          | Docker / TorchScript / ONNX | Docker / ONNX / ML Envs           | Docker / ONNX / TFLite               |
| A/B Testing & Validation | Shadow Deploy / Canary      | Blue-Green / AKS Testing          | Staged Rollout / Traffic Split       |

---

Would you like a companion visual diagram or architecture PDF next?
