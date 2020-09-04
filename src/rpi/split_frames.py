""" 
Networking to stream the camera image and receive commands.
"""
import select
import io
import struct
import sys
import socket

from variables import CurrentState

class SplitFrames(object):
    def __init__(self, socket: socket.socket, connection: io.BufferedRWPair, current_state: CurrentState):
        """Init

        Args:
            socket (socket.socket): Client socket
            connection (io.BufferedRWPair): connection File
            current_state (CurrentState): Current state
        """           
        self.connection = connection
        self.stream = io.BytesIO()
        self.count = 0
        self.end_stream = False
        self.current_state = current_state
        self.socket = socket

    def write(self, buf: bytes):
        """Send image and receive response

        Args:
            buf (bytes): buffer
        """ 
        if buf.startswith(b'\xff\xd8'):
            # Start of new frame; send the old one's length
            # then the data
            size = self.stream.tell()
            if size > 0 and not self.end_stream:
                self.connection.write(struct.pack('<L', size))
                self.connection.flush()
                self.stream.seek(0)
                self.connection.write(self.stream.read(size))
                self.count += 1
                self.stream.seek(0)

                reader, _, _ = select.select([self.socket], [], [], 0)

                if reader:
                    message = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]

                    if message == 8:
                        self.end_stream = True
                        self.current_state.reset_stop_timer()
                    else:
                        self.current_state.set_direction(message)

        self.stream.write(buf)
