docker build -t server -f Dockerfile-server
docker run -p 8081:8081 server