import http.server
import socketserver
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=12880, help="Port number for the HTTP server.")
args = parser.parse_args()

PORT = args.port
ROOT_DIRECTORY = os.path.abspath('.')

class CustomHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIRECTORY, **kwargs)

    def do_GET(self):
        # Serve files from viewer or handle logs
        self.base_path = "/".join(self.path.split('/')[:-1])
        if self.path.startswith('/logs/'):
            self.path = self.path.replace('/logs/', '/logs/')
        elif self.path == '/main.js':
            self.path = '/viewer/main.js'
        elif self.base_path == '' or self.path == '/index.html':
            self.path = self.path.replace('/','/viewer/index.html')
        
        # Serve the file from the calculated path
        return super().do_GET()

# Set up the HTTP server
handler = CustomHttpRequestHandler
with socketserver.TCPServer(("", PORT), handler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()
