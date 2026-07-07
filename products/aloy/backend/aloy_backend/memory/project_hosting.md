---
name: hosting-plan
description: Pori Cloud will be deployed to a cloud provider (EC2 or similar) — user wants to host the API
type: project
---

User plans to host aloy_backend API on EC2 or similar cloud provider. Will need Dockerfile, deployment config, and production setup.

**Why:** aloy_backend is the enterprise backend for Pori — needs to be accessible to the frontend and end users.

**How to apply:** When we get to deployment, set up Docker + a simple deploy pipeline. The DB is already on Supabase (external), so the API server is stateless and easy to containerize.
