import multiprocessing
import subprocess
import time
import logging
import signal
import sys
import os
import psutil
import requests
import websockets
import json
import asyncio
from datetime import datetime
from threading import Thread
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ServerManager:
    def __init__(self):
        self.processes = {}
        self.running = True
        self.health_check_interval = 30  # seconds
        self.restart_delay = 5  # seconds
        self.max_restarts = 3
        self.restart_counts = {}
        self.start_time = time.time()
        self.startup_timeout = 30  # seconds to wait for server startup
        
        # Server configurations
        self.servers = {
            'text_to_sign': {
                'script': 'text_to_sign_server.py',
                'port': 5000,
                'type': 'http',
                'health_endpoint': 'http://localhost:5000/health',
                'description': 'Text to Sign Language Server',
                'startup_delay': 10,
                'host': '0.0.0.0'  # Listen on all interfaces
            },
            'realtime_detection': {
                'script': 'realtime_detection.py',
                'port': 5001,
                'type': 'websocket',
                'health_endpoint': 'ws://localhost:5001',
                'description': 'Real-time Sign Detection Server',
                'startup_delay': 15,
                'host': '0.0.0.0'  # Listen on all interfaces
            }
        }
        
        # Initialize restart counts
        for server_name in self.servers:
            self.restart_counts[server_name] = 0

    def get_network_info(self):
        """Get network interface information"""
        try:
            import socket
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            # Try to get the actual network IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                network_ip = s.getsockname()[0]
                s.close()
            except:
                network_ip = local_ip
                
            return {
                'hostname': hostname,
                'local_ip': local_ip,
                'network_ip': network_ip
            }
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
            return {
                'hostname': 'unknown',
                'local_ip': '127.0.0.1',
                'network_ip': '127.0.0.1'
            }

    def start_server(self, server_name, config):
        """Start a single server process"""
        try:
            logger.info(f"Starting {config['description']}...")
            
            # Set environment variables
            env = os.environ.copy()
            env['PORT'] = str(config['port'])
            env['HOST'] = config.get('host', '0.0.0.0')
            
            # Log the exact command being executed
            logger.info(f"Executing: {sys.executable} {config['script']} with PORT={config['port']}")
            
            process = subprocess.Popen(
                [sys.executable, config['script']],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            self.processes[server_name] = {
                'process': process,
                'config': config,
                'start_time': time.time(),
                'stdout_queue': queue.Queue(),
                'stderr_queue': queue.Queue()
            }
            
            # Start threads to read stdout and stderr
            def read_stdout():
                try:
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        line = line.strip()
                        if line:
                            logger.info(f"[{server_name} STDOUT] {line}")
                except Exception as e:
                    logger.error(f"Error reading stdout for {server_name}: {e}")
            
            def read_stderr():
                try:
                    while True:
                        line = process.stderr.readline()
                        if not line:
                            break
                        line = line.strip()
                        if line:
                            logger.warning(f"[{server_name} STDERR] {line}")
                except Exception as e:
                    logger.error(f"Error reading stderr for {server_name}: {e}")
            
            Thread(target=read_stdout, daemon=True).start()
            Thread(target=read_stderr, daemon=True).start()
            
            logger.info(f"{config['description']} started with PID {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {server_name}: {e}")
            return False

    def stop_server(self, server_name):
        """Stop a single server process"""
        if server_name in self.processes:
            process_info = self.processes[server_name]
            process = process_info['process']
            config = process_info['config']
            
            logger.info(f"Stopping {config['description']}...")
            
            try:
                # Graceful shutdown
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                logger.warning(f"Force killing {config['description']}")
                process.kill()
                process.wait()
            
            del self.processes[server_name]
            logger.info(f"{config['description']} stopped")

    def wait_for_server_startup(self, server_name, config):
        """Wait for server to be ready after startup"""
        logger.info(f"Waiting for {config['description']} to be ready...")
        
        # Wait for the specified startup delay
        logger.info(f"Waiting {config['startup_delay']} seconds for {config['description']} to initialize...")
        
        for i in range(config['startup_delay']):
            time.sleep(1)
            if i > 0 and i % 5 == 0:
                logger.info(f"Still waiting... {i}/{config['startup_delay']} seconds elapsed")
        
        # Then check health with retries
        max_attempts = 8
        retry_delay = 3
        
        for attempt in range(max_attempts):
            logger.info(f"Health check attempt {attempt + 1}/{max_attempts} for {config['description']}")
            
            # Check if port is listening first
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            try:
                result = sock.connect_ex(('localhost', config['port']))
                if result == 0:
                    logger.info(f"Port {config['port']} is listening")
                else:
                    logger.warning(f"Port {config['port']} is not yet listening")
            except Exception as e:
                logger.warning(f"Error checking port {config['port']}: {e}")
            finally:
                sock.close()
            
            if self.check_server_health(server_name, config):
                logger.info(f"{config['description']} is ready")
                return True
            
            if attempt < max_attempts - 1:
                logger.info(f"Health check failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        logger.error(f"{config['description']} failed to become ready within timeout")
        
        # Get final process output for debugging
        if server_name in self.processes:
            logger.info("Attempting to get final process output...")
            try:
                process = self.processes[server_name]['process']
                if process.poll() is None:
                    logger.warning("Process still running but not responding")
                else:
                    logger.error(f"Process exited with code: {process.returncode}")
            except Exception as e:
                logger.error(f"Error checking final process state: {e}")
        
        return False

    async def check_websocket_health(self, uri):
        """Check WebSocket server health with better error handling"""
        try:
            # Increase timeout for initial connection
            async with websockets.connect(uri, ping_timeout=10, open_timeout=10) as websocket:
                # Send ping message
                ping_msg = json.dumps({"type": "ping"})
                await websocket.send(ping_msg)
                
                # Wait for pong response
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(response)
                
                return data.get('type') == 'pong'
        except websockets.exceptions.ConnectionRefused:
            # Server not ready yet
            return False
        except asyncio.TimeoutError:
            logger.warning("WebSocket health check timed out")
            return False
        except Exception as e:
            logger.warning(f"WebSocket health check error: {e}")
            return False

    def check_http_health(self, url):
        """Check HTTP server health with better error handling"""
        try:
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            # Server not ready yet
            return False
        except requests.exceptions.Timeout:
            logger.warning("HTTP health check timed out")
            return False
        except Exception as e:
            logger.warning(f"HTTP health check error: {e}")
            return False

    def check_server_health(self, server_name, config):
        """Check if a server is healthy"""
        if server_name not in self.processes:
            return False
            
        process_info = self.processes[server_name]
        process = process_info['process']
        
        # Check if process is still running
        if process.poll() is not None:
            # Get stderr output for debugging
            try:
                stdout, stderr = process.communicate(timeout=1)
                if stderr:
                    logger.error(f"{config['description']} stderr: {stderr.decode()}")
                if stdout:
                    logger.info(f"{config['description']} stdout: {stdout.decode()}")
            except:
                pass
            logger.warning(f"{config['description']} process has died")
            return False
        
        # Check server responsiveness
        if config['type'] == 'http':
            # Try multiple endpoints for HTTP health check
            health_urls = [
                config['health_endpoint'],
                f"http://localhost:{config['port']}/",
                f"http://localhost:{config['port']}/convert"
            ]
            for url in health_urls:
                if self.check_http_health(url):
                    return True
            logger.warning(f"HTTP health check failed for all endpoints: {health_urls}")
            return False
        elif config['type'] == 'websocket':
            try:
                return asyncio.run(self.check_websocket_health(config['health_endpoint']))
            except Exception as e:
                logger.warning(f"WebSocket health check failed: {e}")
                return False
        
        return True

    def restart_server(self, server_name, config):
        """Restart a server with backoff strategy"""
        if self.restart_counts[server_name] >= self.max_restarts:
            logger.error(f"Max restarts reached for {server_name}. Not restarting.")
            return False
        
        logger.info(f"Restarting {config['description']}...")
        
        # Stop the server
        self.stop_server(server_name)
        
        # Wait before restarting
        time.sleep(self.restart_delay)
        
        # Start the server
        if self.start_server(server_name, config):
            self.restart_counts[server_name] += 1
            logger.info(f"{config['description']} restarted (attempt {self.restart_counts[server_name]})")
            return True
        else:
            logger.error(f"Failed to restart {config['description']}")
            return False

    def monitor_resources(self):
        """Monitor system resources"""
        try:
            # Get system stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Log resource usage
            logger.info(f"System Resources - CPU: {cpu_percent}%, Memory: {memory.percent}%")
            
            # Check individual process resources
            for server_name, process_info in self.processes.items():
                try:
                    pid = process_info['process'].pid
                    proc = psutil.Process(pid)
                    cpu = proc.cpu_percent()
                    mem = proc.memory_info().rss / 1024 / 1024  # MB
                    
                    logger.info(f"{server_name} - CPU: {cpu}%, Memory: {mem:.1f}MB")
                    
                    # Alert if resource usage is too high
                    if cpu > 80 or mem > 1000:  # 1GB
                        logger.warning(f"High resource usage detected for {server_name}")
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")

    def health_check_loop(self):
        """Main health check loop"""
        while self.running:
            try:
                for server_name, config in self.servers.items():
                    if not self.check_server_health(server_name, config):
                        logger.warning(f"{config['description']} health check failed")
                        
                        # Attempt restart
                        if not self.restart_server(server_name, config):
                            logger.error(f"Failed to restart {server_name}. System may be unstable.")
                    else:
                        # Reset restart count on successful health check
                        self.restart_counts[server_name] = 0
                
                # Monitor system resources
                self.monitor_resources()
                
                # Log uptime
                uptime = time.time() - self.start_time
                logger.info(f"Process Manager uptime: {uptime/3600:.1f} hours")
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(10)  # Brief pause before retrying

    def start_all_servers(self):
        """Start all configured servers"""
        logger.info("Starting all servers...")
        
        # Get network information
        network_info = self.get_network_info()
        logger.info(f"Network Information:")
        logger.info(f"  Hostname: {network_info['hostname']}")
        logger.info(f"  Local IP: {network_info['local_ip']}")
        logger.info(f"  Network IP: {network_info['network_ip']}")
        
        # Start servers sequentially to avoid resource conflicts
        for server_name, config in self.servers.items():
            if not self.start_server(server_name, config):
                logger.error(f"Failed to start {server_name}")
                return False
            
            # Wait for this server to be ready before starting the next
            if not self.wait_for_server_startup(server_name, config):
                logger.error(f"{config['description']} failed startup health check")
                return False
        
        logger.info("All servers started successfully and are healthy")
        logger.info("="*60)
        logger.info("SERVER ACCESS INFORMATION:")
        logger.info("="*60)
        
        for server_name, config in self.servers.items():
            port = config['port']
            if config['type'] == 'http':
                logger.info(f"{config['description']}:")
                logger.info(f"  Local access: http://localhost:{port}")
                logger.info(f"  Network access: http://{network_info['network_ip']}:{port}")
                logger.info(f"  Test endpoint: http://{network_info['network_ip']}:{port}/test")
                logger.info(f"  Health check: {config['health_endpoint']}")
            elif config['type'] == 'websocket':
                logger.info(f"{config['description']}:")
                logger.info(f"  Local access: ws://localhost:{port}")
                logger.info(f"  Network access: ws://{network_info['network_ip']}:{port}")
                logger.info(f"  Health check: {config['health_endpoint']}")
            logger.info("-" * 40)
        
        logger.info("="*60)
        return True

    def stop_all_servers(self):
        """Stop all servers gracefully"""
        logger.info("Stopping all servers...")
        self.running = False
        
        for server_name in list(self.processes.keys()):
            self.stop_server(server_name)
        
        logger.info("All servers stopped")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}. Shutting down...")
        self.stop_all_servers()
        sys.exit(0)

    def run(self):
        """Main entry point"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("Process Manager starting...")
        
        # Start all servers
        if not self.start_all_servers():
            logger.error("Failed to start servers. Exiting.")
            sys.exit(1)
        
        # Start health check loop in a separate thread
        health_thread = Thread(target=self.health_check_loop, daemon=True)
        health_thread.start()
        
        logger.info("Process Manager is running. Press Ctrl+C to stop.")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all_servers()

def main():
    """Main function"""
    manager = ServerManager()
    manager.run()

if __name__ == "__main__":
    main()
