### Worker server for running finance jobs
### Receive data / commands from the master server
### When commands are received, they can take a while to compute
### Therefore, commands should always be accompanied by some kind of 
### id. This server will return OK and when the computation is
### comlpete return the results to the server along with the id
### The master server is responsible for taking all these responses
### which can be received in random orders and putting it all together

# curl -X POST -d "param1=poop&param2=pee" http://localhost:8000/post_endpoint

import http.server
import socketserver
import cgi

PORT = 8000

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/post_job':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
    print("serving at port ", PORT)
    httpd.serve_forever()