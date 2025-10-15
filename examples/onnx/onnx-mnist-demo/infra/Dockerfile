# Use nginx to serve static files
FROM nginx:alpine

# Copy the HTML demo and ONNX model to nginx's web directory
COPY cnn_model_webgpu_demo.html /usr/share/nginx/html/index.html
COPY cnn_model.onnx /usr/share/nginx/html/cnn_model.onnx

# Configure nginx to serve .onnx files with the correct MIME type
RUN echo 'types { application/octet-stream onnx; }' > /etc/nginx/conf.d/onnx-mime.conf

# Expose port 8080 (fly.io default)
EXPOSE 8080

# Update nginx to listen on port 8080 instead of 80
RUN sed -i 's/listen       80;/listen       8080;/' /etc/nginx/conf.d/default.conf

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
