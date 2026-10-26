import asyncio
import base64
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import cv2
from PIL import Image
from starlette.datastructures import UploadFile

from app.database.images import Images
from app.routes.general.images import get_image_file
from app.services.database_access import datasets as datasets_db
from app.services.database_access import images as images_db


def _tiff(path: Path, pixels, mode=None) -> bytes:
    image = Image.fromarray(np.asarray(pixels), mode=mode)
    image.save(path, format="TIFF")
    return path.read_bytes()


def _png_pixels(content: bytes):
    with Image.open(io.BytesIO(content)) as image:
        return image.size, image.mode, np.asarray(image).copy()


def _row(image_id: int, full: Path, thumbnail: Path):
    return SimpleNamespace(
        id=image_id,
        dataset_id=4,
        width=3,
        height=2,
        color_mode="RGB",
        description=None,
        file_path=str(full),
        thumbnail_file_path=str(thumbnail),
    )


def _upload(filename: str, content: bytes) -> UploadFile:
    file = tempfile.SpooledTemporaryFile()
    file.write(content)
    file.seek(0)
    return UploadFile(filename=filename, file=file)


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def filter_by(self, **kwargs):
        self.rows = [row for row in self.rows if all(getattr(row, key) == value for key, value in kwargs.items())]
        return self

    def filter(self, *_args):
        return self

    def limit(self, *_args):
        return self

    def first(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return self.rows


class _DB:
    def __init__(self, rows):
        self.rows = rows

    def query(self, *_args):
        return _Query(list(self.rows))


def test_tiff_display_conversion_preserves_dimensions_pixels_and_source(tmp_path):
    pixels = np.array([[0, 1, 257], [4096, 32768, 65535]], dtype=np.uint16)
    source = tmp_path / "source.tiff"
    original = _tiff(source, pixels, mode="I;16")

    preview = images_db.tiff_preview_bytes(source)

    size, mode, actual = _png_pixels(preview)
    assert size == (3, 2)
    assert mode in {"I", "I;16"}
    np.testing.assert_array_equal(actual, pixels.astype(np.int32))
    assert source.read_bytes() == original


def test_big_endian_16_bit_pixels_are_written_as_native_endian_png():
    pixels = np.array([[0, 256], [50000, 65535]], dtype=">u2")
    source = Image.frombytes("I;16B", (2, 2), pixels.tobytes())

    preview = images_db._tiff_png_bytes(source)

    size, mode, actual = _png_pixels(preview)
    assert size == (2, 2)
    assert mode in {"I", "I;16"}
    np.testing.assert_array_equal(actual, pixels.astype(np.int32))


def test_float_preview_uses_full_finite_range_and_rejects_ambiguous_planes(tmp_path):
    source_path = tmp_path / "float.tif"
    source = _tiff(source_path, np.array([[-2, 0], [1, 4]], dtype=np.float32), mode="F")
    preview = images_db.tiff_preview_bytes(source_path)
    _, _, values = _png_pixels(preview)
    np.testing.assert_array_equal(values, np.array([[0, 21845], [32768, 65535]], dtype=np.int32))
    assert source_path.read_bytes() == source

    for pixels in (np.full((2, 2), 3.5, dtype=np.float32),
                   np.array([[0, 1], [np.nan, 2]], dtype=np.float32)):
        with pytest.raises(images_db.TIFFPreviewError):
            images_db._tiff_png_bytes(Image.fromarray(pixels, mode="F"))


def test_upload_keeps_tiff_bytes_and_rejects_multipage_before_writing(tmp_path):
    original_path = tmp_path / "single.tiff"
    original = _tiff(original_path, np.array([[1, 2], [3, 4]], dtype=np.uint16), mode="I;16")
    stored = tmp_path / "data" / "single.tiff"
    thumbnail = tmp_path / "thumbs" / "single.tiff.png"
    upload = _upload("single.tiff", original)

    asyncio.run(images_db.save_image_to_disk(upload, stored, thumbnail))

    assert stored.read_bytes() == original
    with Image.open(stored) as original_image, Image.open(thumbnail) as thumbnail_image:
        assert original_image.size == thumbnail_image.size == (2, 2)
        assert thumbnail_image.format == "PNG"
    assert images_db.tiff_preview_bytes(thumbnail) is None

    first = Image.new("L", (2, 2), 0)
    second = Image.new("L", (2, 2), 255)
    multipage = io.BytesIO()
    first.save(multipage, format="TIFF", save_all=True, append_images=[second])
    rejected_file = tmp_path / "rejected" / "stack.tif"
    rejected_thumbnail = tmp_path / "thumbs" / "stack.tif"
    with pytest.raises(images_db.TIFFPreviewError) as error:
        asyncio.run(images_db.save_image_to_disk(
            _upload("stack.tif", multipage.getvalue()),
            rejected_file,
            rejected_thumbnail,
        ))

    assert "Multi-page TIFF" in error.value.detail
    assert not rejected_file.exists()
    assert not rejected_thumbnail.exists()


def test_corrupt_tiff_upload_and_legacy_display_return_clear_422(tmp_path):
    broken = b"not a TIFF"
    full = tmp_path / "broken.tiff"
    thumbnail = tmp_path / "broken-thumb.tiff"

    with pytest.raises(images_db.TIFFPreviewError) as upload_error:
        asyncio.run(images_db.save_image_to_disk(_upload("broken.tiff", broken), full, thumbnail))
    assert upload_error.value.status_code == 422
    assert "Could not read this TIFF" in upload_error.value.detail
    assert not full.exists() and not thumbnail.exists()

    full.write_bytes(broken)
    thumbnail.write_bytes(broken)
    row = _row(15, full, thumbnail)
    db = _DB([row])
    with pytest.raises(images_db.TIFFPreviewError) as json_error:
        asyncio.run(images_db.get_image_data(15, False, True, db))
    with pytest.raises(images_db.TIFFPreviewError) as file_error:
        asyncio.run(get_image_file(15, db=db, user=None))
    assert json_error.value.status_code == file_error.value.status_code == 422
    assert "Could not read this TIFF" in json_error.value.detail
    assert "Could not read this TIFF" in file_error.value.detail


def test_16_bit_rgb_tiff_is_rejected_before_pillow_clips_channels(tmp_path):
    source = tmp_path / "rgb16.tif"
    pixels = np.arange(12, dtype=np.uint16).reshape((2, 2, 3)) * 5000
    assert cv2.imwrite(str(source), pixels)
    with Image.open(source) as image:
        assert image.mode == "RGB"
        assert max(image.tag_v2[258]) == 16

    with pytest.raises(images_db.TIFFPreviewError, match="Multi-channel high-bit-depth"):
        images_db.tiff_preview_bytes(source)

    rejected_file = tmp_path / "upload" / "rgb16.tif"
    rejected_thumbnail = tmp_path / "thumb" / "rgb16.tif"
    with pytest.raises(images_db.TIFFPreviewError) as error:
        asyncio.run(images_db.save_image_to_disk(
            _upload("rgb16.tif", source.read_bytes()), rejected_file, rejected_thumbnail
        ))
    assert error.value.status_code == 422
    assert not rejected_file.exists() and not rejected_thumbnail.exists()


def test_tiff_upload_records_a_png_thumbnail_path(tmp_path, monkeypatch):
    original = io.BytesIO()
    float_pattern = np.tile(np.array([0, 1], dtype=np.float32), (600, 300))
    Image.fromarray(float_pattern, mode="F").save(original, format="TIFF")
    original_bytes = original.getvalue()
    thumbnail_root = tmp_path / "thumbnails"
    monkeypatch.setattr(images_db, "THUMBNAILS_DIR", str(thumbnail_root))

    async def no_mask(*_args):
        return None

    monkeypatch.setattr(images_db, "create_new_mask", no_mask)

    class InsertDB:
        def add(self, row):
            self.row = row
            row.id = 12

        def flush(self):
            pass

        def commit(self):
            pass

    db = InsertDB()
    asyncio.run(images_db.process_and_save_image(
        _upload("scan.tif", original_bytes), 4, str(tmp_path / "dataset"), db
    ))

    assert Path(db.row.file_path).read_bytes() == original_bytes
    assert Path(db.row.thumbnail_file_path).name == "scan.tif.png"
    with Image.open(db.row.thumbnail_file_path) as thumbnail:
        assert thumbnail.format == "PNG" and thumbnail.size == (500, 500)


def test_tiff_previews_reach_single_batch_dataset_and_file_responses(tmp_path):
    full = tmp_path / "full.tif"
    thumb = tmp_path / "thumb.tif"
    _tiff(full, np.array([[10, 20], [30, 40]], dtype=np.uint8))
    _tiff(thumb, np.array([[5, 15]], dtype=np.uint8))
    row = _row(7, full, thumb)
    db = _DB([row])

    single = asyncio.run(images_db.get_image_data(7, False, True, db))
    thumbnail = asyncio.run(images_db.get_image_data(7, True, True, db))
    batch = asyncio.run(images_db.get_images_data([7], False, True, db))
    batch_thumbnails = asyncio.run(images_db.get_images_data([7], True, True, db))
    dataset = asyncio.run(datasets_db.get_images_of_dataset(4, db, as_base64=True))
    dataset_thumbnails = asyncio.run(
        datasets_db.get_images_of_dataset(4, db, as_thumbnail=True, as_base64=True)
    )

    assert _png_pixels(base64.b64decode(single))[0] == (2, 2)
    assert _png_pixels(base64.b64decode(thumbnail))[0] == (2, 1)
    assert _png_pixels(base64.b64decode(batch[7]))[0] == (2, 2)
    assert _png_pixels(base64.b64decode(batch_thumbnails[7]))[0] == (2, 1)
    assert _png_pixels(base64.b64decode(dataset[7]))[0] == (2, 2)
    assert _png_pixels(base64.b64decode(dataset_thumbnails[7]))[0] == (2, 1)

    file_response = asyncio.run(get_image_file(7, db=db, user=None))
    thumb_response = asyncio.run(get_image_file(7, thumbnail=True, db=db, user=None))
    assert file_response.media_type == thumb_response.media_type == "image/png"
    assert _png_pixels(file_response.body)[0] == (2, 2)
    assert _png_pixels(thumb_response.body)[0] == (2, 1)
    assert "immutable" in file_response.headers["cache-control"]


def test_non_tiff_loaders_and_file_response_keep_existing_behavior(tmp_path):
    full = tmp_path / "ordinary.png"
    thumb = tmp_path / "ordinary-thumb.png"
    Image.new("RGB", (3, 2), (20, 40, 60)).save(full)
    Image.new("RGB", (1, 1), (60, 40, 20)).save(thumb)
    row = _row(9, full, thumb)
    db = _DB([row])
    expected = images_db.ImageModel.from_db(row).load_image(as_base64=True)

    actual = asyncio.run(images_db.get_image_data(9, False, True, db))
    file_response = asyncio.run(get_image_file(9, db=db, user=None))

    assert actual == expected
    assert file_response.path == str(full)
    assert file_response.media_type == "image/png"
    assert "immutable" in file_response.headers["cache-control"]


def test_non_base64_tiff_reads_keep_original_ndarray_contract(tmp_path):
    pixels = np.array([[0, 256, 1000], [4096, 12000, 65535]], dtype=np.uint16)
    full = tmp_path / "scientific.tiff"
    thumb = tmp_path / "scientific-thumb.tiff"
    _tiff(full, pixels, mode="I;16")
    _tiff(thumb, pixels, mode="I;16")
    row = _row(11, full, thumb)
    db = _DB([row])
    expected = images_db.ImageModel.from_db(row).load_image(as_base64=False)

    actual = asyncio.run(images_db.get_image_data(11, False, False, db))
    dataset = asyncio.run(datasets_db.get_images_of_dataset(4, db))

    assert isinstance(actual, np.ndarray)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(dataset[11], expected)
