# Use nginx to serve static files
FROM nginx:alpine

# Copy the HTML demo and ONNX model to nginx's web directory
COPY site/ /usr/share/nginx/html/

# Copy custom nginx configuration
COPY infra/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 8080 (fly.io default)
EXPOSE 8080

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
