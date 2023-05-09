### Master server
### sends jobs to worker servers

### The way I'm planning to use this is that any computational job that needs to be executed should be written
### as a python file and extend the task class
### The master server wont do anything at start, but can begin the computational job by sending it the
### command to do so
### there will also be additional endpoints to get the status of the job, stop it, etc

# curl -X POST -d "param1=poop&param2=pee" http://localhost:8001/post_endpoint
# curl -X POST \
#   -H "Content-Type: application/json" \
#   -d '{"port": 8081}' \
#   http://localhost:5000/register_worker

# import http.server
# import socketserver
# import cgi

# PORT = 8001

# class RequestHandler(http.server.BaseHTTPRequestHandler):
#     def send404(self):
#         self.send_response(404)
#         self.end_headers()
#         self.wfile.write(b"Not Found")
        
#     def do_POST(self):
#         if self.path == '/upload_job':
#             self.send_response(200)
#             self.wfile.write(b"OK")
#         elif self.path == '/post_results':
#             self.send_response(200)
#             self.send_header("Content-type", "text/html")
#             self.end_headers()
#             self.wfile.write(b"OK")
#         else:
#             self.send404()

#     def do_GET(self):
#         if self.path == '/get_jobs':
#             self.send_response(200)
#             self.wfile.write(b"No Jobs loaded")
#         elif self.path == '/start_job':
#             self.send_response(200)
#             self.wfile.write(b"OK")
#         elif self.path == '/stop_job':
#             self.send_response(200)
#             self.wfile.write(b"OK")
#         elif self.path == '/get_node_count':
#             self.send_response(200)
#             self.wfile.write(b"0")
#         elif self.path == "/get_completed_tasks":
#             self.send_response(200)
#             self.wfile.write(b"0")
#         else:
#             self.send404()

# with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
#     print("serving at port ", PORT)
#     httpd.serve_forever()

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

from TestJob import TestJob

class Client():
    def __init__(self, json):
        self.ip = json['ip']
        self.port = json['port']

# Workers
clients = []
jobs = {
    'test_job': TestJob()
}
current_job = 'test_job'

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Define a model for a simple example entity
class Example(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(200))

    def __repr__(self):
        return f'<Example {self.id}>'

# Create the database tables
# db.create_all()

# Routes for CRUD operations
@app.route('/health', methods=['GET'])
def get_health():
    return 'OK'

@app.route('/set_test_data', methods=['GET'])
def set_test_data():
    jobs['test'] = open('TestJob.py', 'r').read()
    clients.append(('127.0.0.1', '8081'))
    
    return 'OK'

@app.route('/get_jobs', methods=['GET'])
def get_jobs():
    return '/n'.join(jobs)

@app.route('/current_job', methods=['GET'])
def get_current_job():
    if current_job == None:
        return 'None'
    else:
        return jobs[current_job].name

@app.route('/start_job', methods=['GET'])
def start_job():
    return 'OK'

@app.route('/stop_job', methods=['GET'])
def stop_job():
    return 'OK'

@app.route('/job_status', methods=['GET'])
def job_status():
    if current_job == None:
        return 'ERR'
    else:
        return 'STOPPED'

@app.route('/next_task', methods=['GET'])
def next_task():
    if current_job == None:
        return 'ERR'
    else:
        return current_job.next()
# @app.route('/examples/<int:id>', methods=['GET'])
# def get_example(id):
#     # Return a JSON representation of the specified example
#     example = Example.query.get_or_404(id)
#     return jsonify(example.__dict__)

# @app.route('/examples/<int:id>', methods=['PUT'])
# def update_example(id):
#     # Update the specified example from JSON data in the request body
#     example = Example.query.get_or_404(id)
#     example_data = request.get_json()
#     example.name = example_data['name']
#     example.description = example_data['description']
#     db.session.commit()
#     return jsonify(example.__dict__)

# @app.route('/examples/<int:id>', methods=['DELETE'])
# def delete_example(id):
#     # Delete the specified example
#     example = Example.query.get_or_404(id)
#     db.session.delete(example)
#     db.session.commit()
#     return '', 204

if __name__ == '__main__':
    app.run(debug=True, port="5000")