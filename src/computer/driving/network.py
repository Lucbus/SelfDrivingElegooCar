"""
The camera stream was build with the example from
https://picamera.readthedocs.io/en/release-1.3/recipes2.html
"""
import socket
import numpy as np
import struct
import io

from variables import Command

class Network():

    def __init__(self, ip: str, port: int) -> None:
        """Inits the network connection

        Args:
            ip (str): Ip to connect to
            port (int): Port to connect to

        Raises:
            ConnectionError: Error if the connection fails
        """        
        print("init network")
        try:
            self.server_socket = socket.socket()
            self.server_socket.bind((ip, port))
            self.server_socket.listen(0)

            self.connection = self.server_socket.accept()[0].makefile('rwb')

        except:
            raise ConnectionError()

    def receive_image(self) -> np.ndarray:
        """Receive an observation from the stream

        Returns:
            np.ndarray: Image of observation
        """        
        image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]

        if not image_len:
            return np.empty

        image_stream = io.BytesIO()
        image_stream.write(self.connection.read(image_len))

        image_stream.seek(0)

        return np.frombuffer(image_stream.getvalue(), dtype=np.uint8)

    def send_command(self, data: Command) -> None:
        """Send command to car.

        Args:
            data (Command): Command
        """        
        data_value = data.value
        self.connection.write(struct.pack('<L', data_value))
        self.connection.flush()

    def close(self) -> None:
        """Closes the current connection
        """        
        self.connection.close()
        self.server_socket.close()