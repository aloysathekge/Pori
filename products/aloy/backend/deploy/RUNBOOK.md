# Pori Cloud — EC2 Deployment Runbook

End-to-end deploy of `aloy_backend` to a single EC2 instance behind nginx +
Let's Encrypt, fronted by `api.pori.aloysathekge.com`. Secrets live in AWS
Systems Manager Parameter Store (SecureString, KMS-encrypted). The client on
Vercel will be pointed at the new API URL at the end.

Run each step in order. Each command is self-contained — copy, edit the
obvious placeholders (`$THING`), run, check the output, then move on.

Region used throughout: **eu-west-1** (colocated with Supabase).

---

## 0. Prereqs (do once, on your laptop)

- AWS CLI installed and `aws configure` done with an admin profile for this account.
- An SSH key pair in `eu-west-1` (create in the EC2 console if you don't have one). Record the key name, e.g. `pori-deploy`.

Check you're in the right AWS account and region:

```bash
aws sts get-caller-identity
aws configure set region eu-west-1
```

---

## 1. Rotate the leaked Supabase DB password

The client repo's `.env` was committing a Postgres password. Rotate it
before putting anything into SSM.

1. Supabase Dashboard → Project → Settings → Database → **Reset database password**.
2. Copy the new password.
3. Delete the `DATABASE_URL` line from `aloy_backend_client/.env` — it never belongs in the frontend.
4. Commit that deletion in the client repo and push.

You'll use the new password in the next step.

---

## 2. Put all secrets in SSM Parameter Store

Build a pooler connection string using the new password (URL-encode any `@`, `:`, `#`, `/`, `?` in the password — use `python3 -c "import urllib.parse;print(urllib.parse.quote('<pw>', safe=''))"`).

Set these env vars locally so the commands below stay tidy:

```bash
export CORS_ORIGINS='https://your-app.vercel.app,http://localhost:5173'
export DATABASE_URL='postgresql+asyncpg://postgres.xxxxxx:URLENCODED_PW@aws-1-eu-west-1.pooler.supabase.com:5432/postgres'
export SUPABASE_URL='https://xxxxxx.supabase.co'
export ANTHROPIC_API_KEY='sk-ant-...'
export GOOGLE_API_KEY='...'
export TAVILY_API_KEY='tvly-...'
# Connections (Gmail/Calendar OAuth). Generate the Fernet key with:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
export CONNECTIONS_ENC_KEY='...'
export GOOGLE_OAUTH_CLIENT_ID='....apps.googleusercontent.com'
export GOOGLE_OAUTH_CLIENT_SECRET='GOCSPX-...'
export BACKEND_BASE_URL='https://api.pori.aloysathekge.com'
export APP_BASE_URL='https://your-app.vercel.app'
# Object storage. 'local' keeps blobs on the instance's /data volume —
# fine for one box. Move to Supabase Storage later with STORAGE_BACKEND=s3
# + the STORAGE_S3_* keys (see .env.example) — config only, no rebuild.
export STORAGE_BACKEND='local'
```

Put each as a SecureString under `/aloy-backend/prod/`:

```bash
for name in DATABASE_URL SUPABASE_URL CORS_ORIGINS ANTHROPIC_API_KEY GOOGLE_API_KEY TAVILY_API_KEY \n            CONNECTIONS_ENC_KEY GOOGLE_OAUTH_CLIENT_ID GOOGLE_OAUTH_CLIENT_SECRET \n            BACKEND_BASE_URL APP_BASE_URL STORAGE_BACKEND; do
  aws ssm put-parameter \
    --name "/aloy-backend/prod/$name" \
    --value "${!name}" \
    --type SecureString \
    --overwrite
done
```

Verify:

```bash
aws ssm get-parameters-by-path --path /aloy-backend/prod --recursive \
  --query 'Parameters[*].Name' --output table
```

---

## 3. Provision the EC2 instance

### 3a. IAM instance profile with SSM read access

```bash
# Trust policy lets EC2 assume the role
cat > /tmp/trust.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON

aws iam create-role --role-name aloy-backend-ec2 \
  --assume-role-policy-document file:///tmp/trust.json

# Permissions: read our SSM prefix + decrypt with the default SSM KMS key
cat > /tmp/pori-ssm.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect":"Allow","Action":["ssm:GetParametersByPath","ssm:GetParameters","ssm:GetParameter"],
     "Resource":"arn:aws:ssm:eu-west-1:*:parameter/aloy-backend/prod/*"},
    {"Effect":"Allow","Action":["kms:Decrypt"],
     "Resource":"*",
     "Condition":{"StringEquals":{"kms:ViaService":"ssm.eu-west-1.amazonaws.com"}}}
  ]
}
JSON

aws iam put-role-policy --role-name aloy-backend-ec2 \
  --policy-name aloy-backend-ssm-read \
  --policy-document file:///tmp/pori-ssm.json

aws iam create-instance-profile --instance-profile-name aloy-backend-ec2
aws iam add-role-to-instance-profile \
  --instance-profile-name aloy-backend-ec2 --role-name aloy-backend-ec2
```

### 3b. Security group

```bash
# Get your current public IP
MY_IP="$(curl -s https://checkip.amazonaws.com)/32"

SG_ID=$(aws ec2 create-security-group \
  --group-name aloy-backend-api \
  --description "Pori Cloud API: 22 from me, 80+443 public" \
  --query GroupId --output text)

aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
  --ip-permissions \
    "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$MY_IP,Description=ssh-me}]" \
    "IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0}]" \
    "IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0}]"

echo "SG_ID=$SG_ID"
```

### 3c. Launch instance

Find the latest Ubuntu 24.04 AMI:

```bash
AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters 'Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*' \
            'Name=state,Values=available' \
  --query 'sort_by(Images, &CreationDate) | [-1].ImageId' --output text)
echo "AMI_ID=$AMI_ID"

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type t3.small \
  --key-name pori-deploy \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name=aloy-backend-ec2 \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=aloy-backend-api}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "INSTANCE_ID=$INSTANCE_ID"

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
```

### 3d. Elastic IP + DNS

```bash
EIP=$(aws ec2 allocate-address --domain vpc --query PublicIp --output text)
aws ec2 associate-address --instance-id "$INSTANCE_ID" --public-ip "$EIP"
echo "ELASTIC_IP=$EIP"
```

In your DNS provider for `aloysathekge.com`:

- Add **A** record: `api.pori` → the `ELASTIC_IP` above. TTL 300.

Verify propagation (may take a few minutes):

```bash
dig +short api.pori.aloysathekge.com
```

---

## 4. Bootstrap the instance

```bash
ssh -i ~/.ssh/pori-deploy.pem ubuntu@api.pori.aloysathekge.com
```

Once in, paste the entire block below in one go:

```bash
set -euxo pipefail

# Docker + compose plugin
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
                        nginx certbot python3-certbot-nginx awscli git
sudo usermod -aG docker ubuntu

# Layout: ONE monorepo clone. The backend lives at products/aloy/backend.
sudo mkdir -p /opt/aloy
sudo chown -R ubuntu:ubuntu /opt/aloy
cd /opt/aloy
git clone https://github.com/aloysathekge/Pori.git Pori

# Confirm SSM access (from the instance role)
aws sts get-caller-identity
AWS_REGION=eu-west-1 /opt/aloy/Pori/products/aloy/backend/deploy/load-env-from-ssm.sh
ls -l /opt/aloy/.env   # should be -rw------- 1 ubuntu

# Log out then back in so the docker group takes effect
exit
```

Reconnect:

```bash
ssh -i ~/.ssh/pori-deploy.pem ubuntu@api.pori.aloysathekge.com
docker ps   # should work without sudo now
```

---

## 5. First app boot + migrations

```bash
cd /opt/aloy/Pori/products/aloy/backend

# Build the image (5-10 min on t3.small first time)
docker compose build

# One-shot: run alembic migrations against the Supabase DB
docker compose run --rm --entrypoint alembic api upgrade head

# Bring the API up
docker compose up -d

# Watch it come healthy
docker compose ps
docker compose logs -f --tail=80
# ^C when you see "Application startup complete" and healthy
```

Internal smoke test (on the box):

```bash
curl -s http://127.0.0.1:8000/v1/health
# → {"status":"ok","version":"0.1"}
```

Two invariants worth knowing (both encoded in the Dockerfile/compose, don't
"fix" them):

- **ONE uvicorn worker.** Live-run re-attach, clarify, stop, and warm resume
  are in-process registries; more API workers break them intermittently.
  Scale the `worker` service (durable-run executor), never the API.
- **`/data` is the durable volume** — object-store blobs (`STORAGE_BACKEND=local`)
  and per-conversation sandbox jails. It's a named volume shared by api+worker;
  `docker compose down` keeps it, `down -v` destroys user files.

---

## 6. nginx + TLS

```bash
# Install the site config and reload nginx (HTTP only at this point)
sudo install -m 0644 /opt/aloy/Pori/products/aloy/backend/deploy/nginx.conf \
  /etc/nginx/sites-available/api.pori.aloysathekge.com
sudo ln -sf /etc/nginx/sites-available/api.pori.aloysathekge.com \
            /etc/nginx/sites-enabled/api.pori.aloysathekge.com
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# Public HTTP check (through nginx)
curl -s http://api.pori.aloysathekge.com/v1/health

# Issue the TLS cert. This edits nginx.conf in place and reloads.
sudo certbot --nginx -d api.pori.aloysathekge.com \
  --agree-tos --non-interactive -m you@example.com --redirect

# Public HTTPS check
curl -s https://api.pori.aloysathekge.com/v1/health
```

Certbot adds its own systemd timer for renewals — no extra work.

---

## 7. Enable the systemd unit (survive reboot)

```bash
sudo chmod +x /opt/aloy/Pori/products/aloy/backend/deploy/load-env-from-ssm.sh
sudo install -m 0644 /opt/aloy/Pori/products/aloy/backend/deploy/aloy-backend.service \
  /etc/systemd/system/aloy-backend.service
sudo systemctl daemon-reload
sudo systemctl enable --now aloy-backend.service
systemctl status aloy-backend.service --no-pager
```

Reboot to confirm it comes up clean:

```bash
sudo reboot
# wait ~30s, reconnect
curl -s https://api.pori.aloysathekge.com/v1/health
```

---

## 8. Point the Vercel client at the new API

1. Vercel Dashboard → `aloy_backend_client` project → Settings → Environment Variables.
2. Add `VITE_API_BASE_URL` = `https://api.pori.aloysathekge.com/v1` for **Production** (and **Preview** if you want previews to hit prod too).
3. Trigger a redeploy (Deployments → latest → Redeploy, or push a commit).
4. Load the Vercel app in a browser. Network tab should show calls going to `api.pori.aloysathekge.com`.

If you see CORS errors, add the exact Vercel origin to `CORS_ORIGINS` in SSM and restart:

```bash
aws ssm put-parameter --name /aloy-backend/prod/CORS_ORIGINS \
  --value 'https://aloy-backend-client.vercel.app,http://localhost:5173' \
  --type SecureString --overwrite

# On the box
sudo systemctl restart aloy-backend.service
```

---

## 9. Day-2 cheatsheet

```bash
# Deploy new code (one repo now)
cd /opt/aloy/Pori && git pull
cd products/aloy/backend
docker compose build && docker compose up -d

# Run new migrations
docker compose run --rm --entrypoint alembic api upgrade head

# Tail logs
docker compose logs -f --tail=200

# Restart everything cleanly
sudo systemctl restart aloy-backend.service

# Rotate a secret
aws ssm put-parameter --name /aloy-backend/prod/ANTHROPIC_API_KEY \
  --value 'sk-ant-new' --type SecureString --overwrite
sudo systemctl restart aloy-backend.service   # pulls the new value via ExecStartPre
```

---

## 10. Moving to `api.pori.dev` later

When you own the new domain:

1. Add an A record `api` → the same Elastic IP.
2. On the box: `sudo certbot --nginx -d api.pori.aloysathekge.com -d api.pori.dev --expand`.
3. Update `CORS_ORIGINS` SSM param if the client moves too. Restart service.

Old hostname keeps working; the cert now covers both.
