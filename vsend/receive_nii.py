#!/usr/bin/env python

import argparse
import os
import sys
from time import sleep

import threading
import socketserver

import vsend.external_image as external_image
import nibabel as nb
import numpy as np


socketserver.TCPServer.allow_reuse_address = True


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def __init__(self, callback, infoclient, *args, **kwargs):
        self.callback = callback
        self.infoclient = infoclient
        super().__init__(*args, **kwargs)

    def handle(self):
        self.callback(self.infoclient, self.request)
        """
        # This commented-out section shows how you might access the current thread and send a response back.
        cur_thread = threading.current_thread()
        response = "{}: {}".format(cur_thread.name, data)
        self.request.sendall(response.encode('utf-8'))  # Ensure string is encoded to bytes for sending
        """


def handler_factory(callback, infoclient):
    def createHandler(*args, **kwargs):
        return ThreadedTCPRequestHandler(callback, infoclient, *args, **kwargs)

    return createHandler


def process_data_callback(infoclient, sock):
    print("received info")
    infoclient.process_data(sock)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


class ImageReceiver:

    def __init__(self, args):
        self.host = args.host
        self.port = args.port

        self._is_running = None
        self._server = None
        self.imagestore = []
        self.save_location = args.save_directory
        self.current_uid = None
        self.current_series_hdr = None
        self.save_4d = args.four_dimensional
        self.stop_after_one_series = args.single_series

        self.ei = external_image.ExternalImage("ExternalImageHeader")

    def stop(self):
        if self._server:
            self._server.shutdown()
        self._is_running = None
        self._server = None

        if self.save_4d:
            self.save_imagestore()

        print("image receiver stopped")

    def start(self):
        self._startserver()

    def check(self):
        if not self._is_running:
            raise RuntimeError("Server is not running")
        return self.imagestore

    def _startserver(self):
        if self._is_running:
            raise RuntimeError("Server already running")

        server = ThreadedTCPServer(
            (self.host, self.port), handler_factory(process_data_callback, self)
        )
        ip, port = server.server_address
        print(f"image receiver running at {ip} on port {port}")
        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        self._is_running = True
        self._server = server

    def process_data(self, sock):
        in_bytes = sock.recv(self.ei.get_header_size())

        if len(in_bytes) != self.ei.get_header_size():
            raise ValueError(
                f"Header data wrong size: expected {self.ei.get_header_size()} bytes, got {len(in_bytes)}"
            )
        print(f"processing {len(in_bytes)} header data bytes")

        hdr = self.ei.process_header(in_bytes)
        print(hdr)

        # validation
        if self.current_uid != hdr.seriesUID:
            self.current_uid = hdr.seriesUID
            self.current_series_hdr = hdr

        img_data = b""
        while len(img_data) < self.ei.get_image_size():
            in_bytes = sock.recv(4096)
            img_data += in_bytes

        img_data = img_data[: self.ei.get_image_size()]

        if len(img_data) != self.ei.get_image_size():
            raise ValueError(
                f"Image data wrong size: expected {self.ei.get_image_size()} bytes, got {len(img_data)}"
            )
        print(f"processing {len(img_data)} image data bytes")

        new_ei = self.ei.process_image(img_data)
        if new_ei:
            if isinstance(new_ei, nb.Nifti1Image) and new_ei not in self.imagestore:
                self.imagestore.append(new_ei)
                if not self.save_4d:
                    self.save_nifti(new_ei)

            if hdr.currentTR + 1 == hdr.totalTR:
                if self.save_4d:
                    self.save_imagestore()
                    self.imagestore = []
                if self.stop_after_one_series:
                    self.stop()
        else:
            self.stop()

    def save_nifti(self, img):
        shape = img.shape
        if len(shape) == 3 or (len(shape) == 4 and shape[3] == 1):
            index = len(self.imagestore) - 1
            imgtype = "pro" if index % 2 == 0 else "ret"
            filename = os.path.join(
                self.save_location,
                f"img-{imgtype}-{self.current_uid}-{index:05d}.nii.gz",
            )
        else:
            filename = os.path.join(
                self.save_location, f"img-{self.current_uid}.nii.gz"
            )
        img.to_filename(filename)
        print(f"Saved to {filename}")

    def save_imagestore(self):
        if not self.imagestore:
            return

        base_shape = self.imagestore[0].shape
        new_shape = base_shape[:3] + (len(self.imagestore),)
        new_data = np.zeros(new_shape)
        for i, img in enumerate(self.imagestore):
            assert img.shape == base_shape, "Mismatched image shapes in imagestore"
            new_data[..., i] = img.get_fdata()

        new_img = nb.Nifti1Image(new_data, self.imagestore[0].affine)
        zooms = (
            self.current_series_hdr.pixelSpacingReadMM,
            self.current_series_hdr.pixelSpacingPhaseMM,
            self.current_series_hdr.pixelSpacingSliceMM,
            self.current_series_hdr.repetitionTimeMS
            + self.current_series_hdr.repetitionDelayMS,
        )
        new_img.header.set_zooms(zooms[: len(new_img.shape)])
        self.save_nifti(new_img)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Run an image receiver.")
    parser.add_argument(
        "-H",
        "--host",
        default="0.0.0.0",
        help="Name of the host to run the image receiver on.",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=50000,
        help="Port to run the image receiver on.",
    )
    parser.add_argument(
        "-d",
        "--save_directory",
        default="./received_data",
        help="Directory to save images to.",
    )
    parser.add_argument(
        "-f",
        "--four_dimensional",
        action="store_true",
        help="Store each image series as a single 4D file.",
    )
    parser.add_argument(
        "-s",
        "--single_series",
        action="store_true",
        help="Shut down the receiver after one entire series has been read.",
    )
    return parser.parse_args(args)


def main(argv):
    args = parse_args(argv[1:])  # Adjust for actual arguments passed to main
    receiver = ImageReceiver(args)
    receiver.start()
    while receiver._is_running:
        sleep(1)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
