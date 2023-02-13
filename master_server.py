### Master server
### sends jobs to worker servers

### The way I'm planning to use this is that any computational job that needs to be executed should be written
### as a python file and extend the task class
### The master server wont do anything at start, but can begin the computational job by sending it the
### command to do so
### there will also be additional endpoints to get the status of the job, stop it, etc

# curl -X POST -d "param1=poop&param2=pee" http://localhost:8001/post_endpoint

import http.server
import socketserver
import cgi

PORT = 8001

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def send404(self):
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")
        
    def do_POST(self):
        if self.path == '/upload_job':
            self.send_response(200)
            self.wfile.write(b"OK")
        elif self.path == '/post_results':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send404()

    def do_GET(self):
        if self.path == '/get_jobs':
            self.send_response(200)
            self.wfile.write(b"No Jobs loaded")
        elif self.path == '/start_job':
            self.send_response(200)
            self.wfile.write(b"OK")
        elif self.path == '/stop_job':
            self.send_response(200)
            self.wfile.write(b"OK")
        elif self.path == '/get_node_count':
            self.send_response(200)
            self.wfile.write(b"0")
        elif self.path == "/get_completed_tasks":
            self.send_response(200)
            self.wfile.write(b"0")
        else:
            self.send404()

with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
    print("serving at port ", PORT)
    httpd.serve_forever()