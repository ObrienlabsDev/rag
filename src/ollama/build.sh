docker build -t ollama-http .
docker stop ollama-http
docker rm ollama-http
docker run -d -p 8080:80 --name ollama-http ollama-http
