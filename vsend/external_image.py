from collections import namedtuple
import struct
from typing import Any

import nibabel as nb
import numpy as np


def mosaic(data):
    x, y, z = data.shape
    n = int(np.ceil(np.sqrt(z)))
    X = np.zeros((n * x, n * y), dtype=data.dtype)
    for idx in range(z):
        x_idx = int(np.floor(idx / n)) * x
        y_idx = (idx % n) * y
        X[x_idx : x_idx + x, y_idx : y_idx + y] = data[..., idx]
    return X


def demosaic(mosaic, x, y, z):
    data = np.zeros((x, y, z), dtype=mosaic.dtype)
    n = int(np.ceil(np.sqrt(z)))
    dim = int(np.sqrt(np.prod(mosaic.shape)))
    mosaic = mosaic.reshape(dim, dim)
    for idx in range(z):
        x_idx = int(np.floor(idx / n)) * x
        y_idx = (idx % n) * y
        data[..., idx] = mosaic[x_idx : x_idx + x, y_idx : y_idx + y]
    return data


class ExternalImage(object):
    # Definition of the Python equivalent of the C++ openheader datastructure.
    # See src/io/RtExternalImageInfo.h
    struct_def = (
        ("magic", "5s"),
        ("headerVersion", "i"),
        ("seriesUID", "64s"),
        ("scanType", "64s"),
        ("imageType", "16s"),
        ("note", "256s"),
        ("dataType", "16s"),
        ("isLittleEndian", "?"),
        ("isMosaic", "?"),
        ("pixelSpacingReadMM", "d"),
        ("pixelSpacingPhaseMM", "d"),
        ("pixelSpacingSliceMM", "d"),
        ("sliceGapMM", "d"),
        ("numPixelsRead", "i"),
        ("numPixelsPhase", "i"),
        ("numSlices", "i"),
        ("voxelToWorldMatrix", "16f"),
        ("repetitionTimeMS", "i"),
        ("repetitionDelayMS", "i"),
        ("currentTR", "i"),
        ("totalTR", "i"),
        ("isMotionCorrected", "?"),
        ("mcOrder", "5s"),
        ("mcTranslationXMM", "d"),
        ("mcTranslationYMM", "d"),
        ("mcTranslationZMM", "d"),
        ("mcRotationXRAD", "d"),
        ("mcRotationYRAD", "d"),
        ("mcRotationZRAD", "d"),
    )

    def __init__(
        self, typename: str, format_def: tuple[tuple[str, str], ...] = struct_def
    ) -> None:
        self.names = [name for name, _ in format_def]
        self.formatstr = "".join(fmt for _, fmt in format_def)
        self.header_fmt = struct.Struct(self.formatstr)
        self.named_tuple_class = namedtuple(typename, self.names)
        self.hdr = None
        self.img = None
        self.num_bytes: int | None = None

    def hdr_from_bytes(self, byte_str: bytes) -> Any:
        """
        Converts a bytes object into a namedtuple based on the header format.

        Args:
            byte_str (bytes): The binary data to be unpacked into the header structure.

        Returns:
            Any: An instance of the namedtuple class representing the unpacked header.
        """
        alist = list(self.header_fmt.unpack(byte_str))
        values = []
        for _, key in enumerate(self.names):
            if key != "voxelToWorldMatrix":
                val = alist.pop(0)
                if isinstance(val, bytes):
                    values.append(
                        val.split(b"\0", 1)[0].decode("utf-8")
                    )  # Decode bytes to string
                else:
                    values.append(val)
            else:
                values.append([alist.pop(0) for _ in range(16)])
        return self.named_tuple_class._make(tuple(values))

    def hdr_to_bytes(self, hdr_info: Any) -> bytes:
        """
        Packs the header information back into a bytes object based on the header format.

        Args:
            hdr_info (Any): An instance of the namedtuple containing the header information to be packed.

        Returns:
            bytes: The packed binary data.
        """
        values = []
        for val in hdr_info._asdict().values():
            if isinstance(val, list):
                values.extend(val)
            else:
                if isinstance(val, str):
                    val = val.encode("utf-8")
                values.append(val)
        return self.header_fmt.pack(*values)

    def create_header(self, img, idx, nt, mosaic: bool):
        x, y, z, t = img.shape
        # Updated to direct attribute access if using a newer nibabel version
        sx, sy, sz, tr = img.header.get_zooms()
        # Direct access to the affine attribute
        affine = img.affine.flatten().tolist()

        EInfo = self.named_tuple_class
        infotuple = EInfo(
            magic=b"ERTI",  # Keep as bytes since it's likely intended to be binary data
            headerVersion=1,
            seriesUID="someuid".encode(
                "ascii"
            ),  # Encode to bytes if required for binary compatibility
            scanType="EPI".encode(
                "ascii"
            ),  # Encode strings intended for binary data fields
            imageType="3D".encode("ascii"),  # Encode strings
            note="some note to leave".encode("ascii"),  # Encode strings
            dataType="int16_t".encode("ascii"),  # Encode strings
            isLittleEndian=True,
            isMosaic=mosaic,
            pixelSpacingReadMM=sx,
            pixelSpacingPhaseMM=sy,
            pixelSpacingSliceMM=sz,
            sliceGapMM=0.0,
            numPixelsRead=x,
            numPixelsPhase=y,
            numSlices=z,
            voxelToWorldMatrix=affine,  # Already a list of floats, no need to encode
            repetitionTimeMS=int(
                tr * 1000
            ),  # Ensure it's an int if `tr` could be a float
            repetitionDelayMS=0,
            currentTR=idx,
            totalTR=nt,
            isMotionCorrected=True,
            mcOrder=b"XYZT",  # Bytes, assuming this is meant to be binary data
            mcTranslationXMM=0.1,
            mcTranslationYMM=0.2,
            mcTranslationZMM=0.01,
            mcRotationXRAD=0.001,
            mcRotationYRAD=0.002,
            mcRotationZRAD=0.0001,
        )

        return infotuple

    def get_header_size(self):
        return self.header_fmt.size

    def get_image_size(self):
        return self.num_bytes

    def from_image(self, img, idx, nt, mosaic=True):
        hdrinfo = self.create_header(img, idx, nt, mosaic)
        # Update to use get_fdata for compatibility with recent nibabel versions.
        if idx is not None:
            data = img.get_fdata()[..., idx]
        else:
            data = img.get_fdata()
        if mosaic:
            data = mosaic(data)
        data = (
            data.astype(np.uint16).flatten().tolist()
        )  # Ensure data is uint16 for struct.pack compatibility
        num_elem = len(data)
        return self.hdr_to_bytes(hdrinfo), struct.pack(f"{num_elem}H", *data)

    def make_img(self, in_bytes):
        h = self.hdr
        if h.dataType != "int16_t":  # Ensure dataType is decoded from bytes to str
            raise ValueError(f"Unsupported data type: {h.dataType}")

        # Correct unpacking considering the actual byte size
        data = struct.unpack(f"{self.num_bytes // 2}H", in_bytes)
        if h.isMosaic:
            data = demosaic(
                np.array(data, dtype=np.uint16),
                h.numPixelsRead,
                h.numPixelsPhase,
                h.numSlices,
            )
        else:
            data = np.array(data, dtype=np.uint16).reshape(
                (h.numPixelsRead, h.numPixelsPhase, h.numSlices)
            )
        affine = np.array(h.voxelToWorldMatrix).reshape((4, 4))
        img = nb.Nifti1Image(data, affine)
        img.header.set_zooms(
            (h.pixelSpacingReadMM, h.pixelSpacingPhaseMM, h.pixelSpacingSliceMM)
        )
        img.header.set_xyzt_units("mm", "msec")
        return img

    def process_header(self, in_bytes):
        magic = struct.unpack("4s", in_bytes[:4])[0]
        print(magic.decode("utf-8"))
        if magic in [b"ERTI", b"SIMU"]:
            self.hdr = self.hdr_from_bytes(in_bytes)
            print(f"header received: TR={self.hdr.currentTR}")
            if self.hdr.isMosaic:
                nrows = int(np.ceil(np.sqrt(self.hdr.numSlices)))
                self.num_bytes = (
                    2 * self.hdr.numPixelsRead * self.hdr.numPixelsPhase * nrows * nrows
                )
            else:
                self.num_bytes = (
                    2
                    * self.hdr.numPixelsRead
                    * self.hdr.numPixelsPhase
                    * self.hdr.numSlices
                )
            print(f"Requires: {self.num_bytes} bytes")
            return self.hdr
        else:
            raise ValueError(f"Unknown magic number {magic.decode('utf-8')}")

    def process_image(self, in_bytes):
        self.img = self.make_img(in_bytes)
        return self.img
