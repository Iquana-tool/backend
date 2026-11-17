"""Robustness checks for TIFF formats beyond the core preview examples."""

import asyncio
import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, features
from starlette.datastructures import UploadFile

from app.routes.general.images import get_image_file
from app.services.database_access import images as images_db


def _upload(filename: str, content: bytes) -> UploadFile:
    file = tempfile.SpooledTemporaryFile()
    file.write(content)
    file.seek(0)
    return UploadFile(filename=filename, file=file)


def _tiff_bytes(array: np.ndarray, compression: str = "raw") -> bytes:
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="TIFF", compression=compression)
    return output.getvalue()


def _png_contents(content: bytes) -> tuple[str, np.ndarray]:
    with Image.open(io.BytesIO(content)) as image:
        assert image.format == "PNG"
        return image.mode, np.asarray(image).copy()


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def filter_by(self, **kwargs):
        self.rows = [row for row in self.rows if all(getattr(row, key) == value for key, value in kwargs.items())]
        return self

    def filter(self, *_args):
        return self

    def first(self):
        return self.rows[0] if self.rows else None


class _UploadDB:
    def __init__(self):
        self.rows = []

    def add(self, row):
        row.id = len(self.rows) + 1
        self.rows.append(row)

    def flush(self):
        pass

    def commit(self):
        pass

    def query(self, *_args):
        return _Query(list(self.rows))


@pytest.mark.parametrize(
    "mode, pixels",
    [
        ("L", np.array([[0, 16, 255], [40, 128, 220]], dtype=np.uint8)),
        ("RGB", np.array([[[1, 2, 3], [40, 50, 60]], [[100, 110, 120], [200, 210, 220]]], dtype=np.uint8)),
        (
            "RGBA",
            np.array(
                [[[1, 2, 3, 0], [40, 50, 60, 64]], [[100, 110, 120, 128], [200, 210, 220, 255]]],
                dtype=np.uint8,
            ),
        ),
    ],
)
@pytest.mark.parametrize("compression", ["raw", "tiff_lzw", "tiff_adobe_deflate"])
def test_uint8_tiff_modes_and_compressions_survive_upload_and_browser_reads(
    tmp_path, monkeypatch, mode, pixels, compression
):
    if compression != "raw" and not features.check("libtiff"):
        pytest.skip(f"Pillow was built without libtiff support for {compression}")

    original = _tiff_bytes(pixels, compression)
    expected_compression_tag = {"raw": 1, "tiff_lzw": 5, "tiff_adobe_deflate": 8}[compression]
    with Image.open(io.BytesIO(original)) as uploaded:
        assert uploaded.tag_v2[259] == expected_compression_tag

    thumbnail_root = tmp_path / "thumbnails"
    monkeypatch.setattr(images_db, "THUMBNAILS_DIR", str(thumbnail_root))

    async def no_mask(*_args):
        return None

    monkeypatch.setattr(images_db, "create_new_mask", no_mask)
    db = _UploadDB()
    image_id = asyncio.run(
        images_db.process_and_save_image(
            _upload("matrix.TIF", original), 1, str(tmp_path / "dataset"), db
        )
    )
    row = db.rows[0]
    source = Path(row.file_path)
    assert image_id == row.id == 1
    assert source.suffix == ".TIF"
    assert source.read_bytes() == original

    response = asyncio.run(get_image_file(image_id, thumbnail=False, db=db, user=None))
    assert response.media_type == "image/png"
    mode_out, pixels_out = _png_contents(Path(response.path).read_bytes())
    assert mode_out == mode
    np.testing.assert_array_equal(pixels_out, pixels)

    encoded = asyncio.run(images_db.get_image_data(image_id, False, True, db))
    mode_b64, pixels_b64 = _png_contents(base64.b64decode(encoded))
    assert mode_b64 == mode
    np.testing.assert_array_equal(pixels_b64, pixels)

    thumbnail_response = asyncio.run(get_image_file(image_id, thumbnail=True, db=db, user=None))
    assert thumbnail_response.media_type == "image/png"
    thumbnail_mode, thumbnail_pixels = _png_contents(Path(thumbnail_response.path).read_bytes())
    assert thumbnail_mode == mode
    np.testing.assert_array_equal(thumbnail_pixels, pixels)

    encoded_thumbnail = asyncio.run(images_db.get_image_data(image_id, True, True, db))
    thumbnail_mode, thumbnail_pixels = _png_contents(base64.b64decode(encoded_thumbnail))
    assert thumbnail_mode == mode
    np.testing.assert_array_equal(thumbnail_pixels, pixels)


@pytest.mark.parametrize("byteorder, expected_mode", [("<", "I;16"), (">", "I;16B")])
def test_real_uint16_tiffs_in_both_byte_orders_keep_12_and_16_bit_range(tmp_path, byteorder, expected_mode):
    tifffile = pytest.importorskip("tifffile")
    pixels = np.array([[0, 4095], [50000, 65535]], dtype=np.uint16)
    encoded = io.BytesIO()
    tifffile.imwrite(encoded, pixels, byteorder=byteorder, photometric="minisblack")
    original = encoded.getvalue()
    source = tmp_path / f"uint16-{byteorder == '>'}.tif"
    source.write_bytes(original)

    with Image.open(source) as image:
        assert image.format == "TIFF"
        assert image.mode == expected_mode

    preview_path = images_db.tiff_preview_file(source)
    assert preview_path is not None
    mode, actual = _png_contents(preview_path.read_bytes())
    assert mode == "L"
    np.testing.assert_array_equal(actual, np.array([[0, 16], [195, 255]], dtype=np.uint8))
    assert source.read_bytes() == original


def test_bigtiff_uint16_source_is_read_and_preserved(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    pixels = np.array([[0, 256], [50000, 65535]], dtype=np.uint16)
    encoded = io.BytesIO()
    tifffile.imwrite(encoded, pixels, bigtiff=True, photometric="minisblack")
    original = encoded.getvalue()
    assert original[2:4] in (b"+\x00", b"\x00+")

    source = tmp_path / "big.tiff"
    source.write_bytes(original)
    preview = images_db.tiff_preview_bytes(source)

    mode, actual = _png_contents(preview)
    assert mode == "L"
    np.testing.assert_array_equal(actual, np.array([[0, 1], [195, 255]], dtype=np.uint8))
    assert source.read_bytes() == original


def test_large_tiled_uint16_upload_keeps_native_size_and_aspect_ratio(tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    pixels = (np.arange(2048 * 1536, dtype=np.uint32).reshape((1536, 2048)) % 4096).astype(np.uint16)
    encoded = io.BytesIO()
    tifffile.imwrite(encoded, pixels, tile=(256, 256), photometric="minisblack")
    original = encoded.getvalue()
    with tifffile.TiffFile(io.BytesIO(original)) as tiff:
        assert tiff.pages[0].is_tiled

    monkeypatch.setattr(images_db, "THUMBNAILS_DIR", str(tmp_path / "thumbnails"))

    async def no_mask(*_args):
        return None

    monkeypatch.setattr(images_db, "create_new_mask", no_mask)
    db = _UploadDB()
    image_id = asyncio.run(
        images_db.process_and_save_image(
            _upload("large.tiff", original), 1, str(tmp_path / "dataset"), db
        )
    )
    row = db.rows[0]
    source = Path(row.file_path)
    assert (row.width, row.height) == (2048, 1536)
    assert source.read_bytes() == original

    full_response = asyncio.run(get_image_file(image_id, thumbnail=False, db=db, user=None))
    assert full_response.media_type == "image/png"
    full_mode, full_pixels = _png_contents(Path(full_response.path).read_bytes())
    assert full_mode == "L"
    assert full_pixels.shape == (1536, 2048)
    assert full_pixels.min() == 0 and full_pixels.max() == 255

    thumbnail_response = asyncio.run(get_image_file(image_id, thumbnail=True, db=db, user=None))
    assert thumbnail_response.media_type == "image/png"
    thumbnail_mode, thumbnail_pixels = _png_contents(Path(thumbnail_response.path).read_bytes())
    assert thumbnail_mode == "L"
    assert thumbnail_pixels.shape == (375, 500)
    assert max(thumbnail_pixels.shape) <= 500
    assert thumbnail_pixels.shape[1] / thumbnail_pixels.shape[0] == pytest.approx(4 / 3)

    full_base64 = asyncio.run(images_db.get_image_data(image_id, False, True, db))
    thumb_base64 = asyncio.run(images_db.get_image_data(image_id, True, True, db))
    assert base64.b64decode(full_base64) == Path(full_response.path).read_bytes()
    assert base64.b64decode(thumb_base64) == Path(thumbnail_response.path).read_bytes()
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "pixels",
    [
        np.full((2, 2), 3.5, dtype=np.float32),
        np.array([[0, 1], [np.nan, 2]], dtype=np.float32),
        np.array([[0, 1], [np.inf, 2]], dtype=np.float32),
    ],
)
def test_ambiguous_float_tiff_uploads_fail_before_creating_files(tmp_path, pixels):
    output = io.BytesIO()
    Image.fromarray(pixels, mode="F").save(output, format="TIFF")
    source = tmp_path / "images" / "invalid.tif"
    thumbnail = tmp_path / "thumbnails" / "invalid.tif.png"

    with pytest.raises(images_db.TIFFPreviewError) as error:
        asyncio.run(images_db.save_image_to_disk(_upload("invalid.tif", output.getvalue()), source, thumbnail))

    assert error.value.status_code == 422
    assert not source.exists()
    assert not thumbnail.exists()


def test_truncated_tiff_with_misleading_suffix_fails_upload_before_writing(tmp_path):
    pixels = np.arange(128 * 128, dtype=np.uint8).reshape((128, 128))
    truncated = _tiff_bytes(pixels)[:-1]
    source = tmp_path / "images" / "truncated.jpg"
    thumbnail = tmp_path / "thumbnails" / "truncated.jpg"

    with pytest.raises(images_db.TIFFPreviewError) as error:
        asyncio.run(images_db.save_image_to_disk(_upload("truncated.jpg", truncated), source, thumbnail))

    assert error.value.status_code == 422
    assert not source.exists()
    assert not thumbnail.exists()
