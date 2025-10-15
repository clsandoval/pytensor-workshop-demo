# Deploying PyTensor CNN Demo to fly.io

## Prerequisites

1. Install the fly.io CLI:
   ```bash
   # On Windows (PowerShell)
   iwr https://fly.io/install.ps1 -useb | iex

   # On macOS/Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. Sign up or log in to fly.io:
   ```bash
   fly auth signup  # or fly auth login
   ```

## Deployment Steps

1. **Navigate to the examples/onnx directory:**
   ```bash
   cd examples/onnx
   ```

2. **Launch the app (first time):**
   ```bash
   fly launch
   ```

   - It will detect the Dockerfile automatically
   - When asked, choose a unique app name (or keep the suggested one)
   - Choose your preferred region (e.g., `iad` for US East)
   - Say **NO** to PostgreSQL and Redis when asked
   - Say **YES** to deploy now

3. **Deploy updates later:**
   ```bash
   fly deploy
   ```

4. **Open your deployed app:**
   ```bash
   fly open
   ```

## What the Dockerfile Does

The Dockerfile:
- Uses **nginx** (lightweight web server) to serve static files
- Copies `cnn_model_webgpu_demo.html` as the main page
- **Copies `cnn_model.onnx`** to the web directory (so the browser can load it)
- Configures the correct MIME type for `.onnx` files
- Runs on port 8080 (fly.io standard)

## Troubleshooting

**View logs:**
```bash
fly logs
```

**Check app status:**
```bash
fly status
```

**SSH into the running container:**
```bash
fly ssh console
```

**Test locally with Docker:**
```bash
docker build -t pytensor-demo .
docker run -p 8080:8080 pytensor-demo
# Then visit http://localhost:8080
```

## Cost

- fly.io offers a generous free tier
- This app uses minimal resources (256MB RAM, 1 shared CPU)
- With `auto_stop_machines = true`, it stops when not in use
- You won't be charged for idle time

## Custom Domain (Optional)

To add a custom domain:
```bash
fly certs create yourdomain.com
```

Then add a DNS record as instructed by fly.io.
