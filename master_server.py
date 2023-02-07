### Master server
### sends jobs to worker servers

# curl -X POST -d "param1=poop&param2=pee" http://localhost:8001/post_endpoint

import http.server
import socketserver
import cgi

PORT = 8001

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/post_results':
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