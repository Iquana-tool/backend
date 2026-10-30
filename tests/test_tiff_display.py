import asyncio
import base64
import io
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import cv2
from PIL import Image
from starlette.datastructures import UploadFile

from app.database.images import Images
from app.routes.general.images import get_image_file, upload_image, upload_images
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
    assert mode == "L"
    np.testing.assert_array_equal(actual, np.array([[0, 0, 1], [16, 128, 255]], dtype=np.uint8))
    assert source.read_bytes() == original


def test_big_endian_16_bit_pixels_are_stretched_to_png():
    pixels = np.array([[0, 256], [50000, 65535]], dtype=">u2")
    source = Image.frombytes("I;16B", (2, 2), pixels.tobytes())

    preview = images_db._tiff_png_bytes(source)

    size, mode, actual = _png_pixels(preview)
    assert size == (2, 2)
    assert mode == "L"
    np.testing.assert_array_equal(actual, np.array([[0, 1], [195, 255]], dtype=np.uint8))


def test_integer_tiff_preview_stretches_12_bit_range_and_maps_constant_to_black(tmp_path):
    pixels = np.array([[100, 500], [1200, 4095]], dtype=np.uint16)
    source = tmp_path / "12-bit.tif"
    original = _tiff(source, pixels, mode="I;16")

    _, mode, actual = _png_pixels(images_db.tiff_preview_bytes(source))

    assert mode == "L"
    np.testing.assert_array_equal(actual, np.array([[0, 26], [70, 255]], dtype=np.uint8))
    assert source.read_bytes() == original

    constant = Image.fromarray(np.full((2, 3), 4095, dtype=np.uint16), mode="I;16")
    _, mode, actual = _png_pixels(images_db._tiff_png_bytes(constant))
    assert mode == "L"
    np.testing.assert_array_equal(actual, np.zeros((2, 3), dtype=np.uint8))


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
    assert _png_pixels(Path(file_response.path).read_bytes())[0] == (2, 2)
    assert _png_pixels(Path(thumb_response.path).read_bytes())[0] == (2, 1)
    assert "immutable" in file_response.headers["cache-control"]


def test_legacy_16_bit_png_thumbnail_uses_tiff_display_range(tmp_path):
    full = tmp_path / "legacy-source.tif"
    thumbnail = tmp_path / "legacy-source.tif.png"
    pixels = np.array([[0, 256], [50000, 65535]], dtype=np.uint16)
    _tiff(full, pixels, mode="I;16")
    Image.fromarray(pixels, mode="I;16").save(thumbnail, format="PNG")
    original_full = full.read_bytes()
    row = _row(8, full, thumbnail)

    encoded = asyncio.run(images_db.get_image_data(8, True, True, _DB([row])))
    file_response = asyncio.run(get_image_file(8, thumbnail=True, db=_DB([row]), user=None))
    encoded_preview = _png_pixels(base64.b64decode(encoded))
    file_preview = _png_pixels(Path(file_response.path).read_bytes())

    for size, mode, values in (encoded_preview, file_preview):
        assert size == (2, 2)
        assert mode == "L"
        assert values.min() == 0 and values.max() == 255
    assert file_response.media_type == "image/png"
    assert base64.b64decode(encoded) == Path(file_response.path).read_bytes()
    assert full.read_bytes() == original_full


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


def _multipage_tiff_bytes() -> bytes:
    first = Image.new("L", (2, 2), 0)
    second = Image.new("L", (2, 2), 255)
    output = io.BytesIO()
    first.save(output, format="TIFF", save_all=True, append_images=[second])
    return output.getvalue()


def test_batch_and_dataset_display_skip_unsupported_tiff_but_keep_valid_images(tmp_path):
    valid_full = tmp_path / "valid.tif"
    valid_thumb = tmp_path / "valid-thumb.tif"
    _tiff(valid_full, np.array([[0, 4095]], dtype=np.uint16), mode="I;16")
    _tiff(valid_thumb, np.array([[100, 200]], dtype=np.uint16), mode="I;16")

    unsupported = _multipage_tiff_bytes()
    bad_full = tmp_path / "multipage.tif"
    bad_thumb = tmp_path / "multipage-thumb.tif"
    bad_full.write_bytes(unsupported)
    bad_thumb.write_bytes(unsupported)
    db = _DB([_row(1, valid_full, valid_thumb), _row(2, bad_full, bad_thumb)])

    batch = asyncio.run(images_db.get_images_data([1, 2], False, True, db))
    batch_thumbnails = asyncio.run(images_db.get_images_data([1, 2], True, True, db))
    dataset = asyncio.run(datasets_db.get_images_of_dataset(4, db, as_base64=True))
    dataset_thumbnails = asyncio.run(
        datasets_db.get_images_of_dataset(4, db, as_thumbnail=True, as_base64=True)
    )

    for result in (batch, batch_thumbnails, dataset, dataset_thumbnails):
        assert set(result) == {1}
        size, mode, pixels = _png_pixels(base64.b64decode(result[1]))
        assert size == (2, 1)
        assert mode == "L"
        assert pixels.min() == 0 and pixels.max() == 255


def test_upload_multi_preflights_every_tiff_before_saving_any_file(tmp_path, monkeypatch):
    valid = _upload("valid.tif", _tiff(
        tmp_path / "upload.tif", np.array([[0, 1], [2, 3]], dtype=np.uint16), mode="I;16"
    ))
    unsupported = _upload("multipage.tif", _multipage_tiff_bytes())
    saves = []

    async def get_dataset(*_args, **_kwargs):
        return SimpleNamespace(folder_path=str(tmp_path / "dataset"))

    async def process_and_save(*_args, **_kwargs):
        saves.append(True)
        return len(saves)

    monkeypatch.setattr(datasets_db, "get_dataset", get_dataset)
    monkeypatch.setattr(images_db, "process_and_save_image", process_and_save)

    with pytest.raises(images_db.TIFFPreviewError) as error:
        asyncio.run(upload_images(4, [valid, unsupported], db=object(), user=None))

    assert error.value.status_code == 422
    assert "Multi-page TIFF" in error.value.detail
    assert saves == []


def test_non_tiff_suffix_returns_without_opening_with_pillow(tmp_path, monkeypatch):
    non_tiff = tmp_path / "broken.png"
    thumbnail = tmp_path / "broken-thumb.png"
    non_tiff.write_bytes(b"not an image")
    thumbnail.write_bytes(b"not an image")

    def unexpected_open(*_args, **_kwargs):
        pytest.fail("non-TIFF suffix should bypass Pillow preview decoding")

    monkeypatch.setattr(images_db.Image, "open", unexpected_open)
    assert images_db.tiff_preview_file(non_tiff) is None
    assert images_db.tiff_preview_bytes(non_tiff) is None
    response = asyncio.run(get_image_file(17, db=_DB([_row(17, non_tiff, thumbnail)]), user=None))
    assert Path(response.path) == non_tiff


def test_decompression_bomb_is_422_and_does_not_poison_mixed_batch(tmp_path, monkeypatch):
    good_full = tmp_path / "good.tif"
    good_thumb = tmp_path / "good-thumb.tif"
    huge_full = tmp_path / "huge.tif"
    huge_thumb = tmp_path / "huge-thumb.tif"
    _tiff(good_full, np.array([[0, 100], [200, 300]], dtype=np.uint16), mode="I;16")
    huge_bytes = _tiff(huge_full, np.arange(9, dtype=np.uint16).reshape(3, 3), mode="I;16")
    _tiff(good_thumb, np.array([[0, 1], [2, 3]], dtype=np.uint8))
    _tiff(huge_thumb, np.arange(9, dtype=np.uint8).reshape(3, 3))
    db = _DB([_row(1, good_full, good_thumb), _row(2, huge_full, huge_thumb)])
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 4)

    with pytest.raises(images_db.TIFFPreviewError) as direct_error:
        asyncio.run(images_db.get_image_data(2, False, True, db))
    assert direct_error.value.status_code == 422
    assert "Could not read" in direct_error.value.detail

    async def get_dataset(*_args, **_kwargs):
        return SimpleNamespace(folder_path=str(tmp_path / "upload-dataset"))

    monkeypatch.setattr(datasets_db, "get_dataset", get_dataset)
    with pytest.raises(images_db.TIFFPreviewError) as upload_error:
        asyncio.run(upload_image(4, _upload("huge.tif", huge_bytes), db=object(), user=None))
    assert upload_error.value.status_code == 422
    assert not (tmp_path / "upload-dataset" / "images" / "huge.tif").exists()

    batch = asyncio.run(images_db.get_images_data([1, 2], False, True, db))
    assert set(batch) == {1}
    assert _png_pixels(base64.b64decode(batch[1]))[0] == (2, 2)


def test_tiff_file_and_base64_share_cache_and_source_change_invalidates_it(tmp_path, monkeypatch):
    full = tmp_path / "images" / "image.tiff"
    thumbnail = tmp_path / "thumbnails" / "image.tiff.png"
    full.parent.mkdir()
    thumbnail.parent.mkdir()
    _tiff(full, np.array([[0, 100], [200, 300]], dtype=np.uint16), mode="I;16")
    thumbnail.write_bytes(b"thumbnail placeholder")
    row = _row(21, full, thumbnail)

    conversion_count = 0
    original_convert = images_db._tiff_png_bytes

    def count_conversion(image):
        nonlocal conversion_count
        conversion_count += 1
        return original_convert(image)

    monkeypatch.setattr(images_db, "_tiff_png_bytes", count_conversion)

    preview_path = images_db.tiff_preview_file(full, thumbnail)
    file_response = asyncio.run(get_image_file(21, db=_DB([row]), user=None))
    encoded = asyncio.run(images_db.get_image_data(21, False, True, _DB([row])))

    assert Path(file_response.path) == preview_path
    assert preview_path.parent == thumbnail.parent
    assert base64.b64decode(encoded) == preview_path.read_bytes()
    assert conversion_count == 1

    old_bytes = preview_path.read_bytes()
    original_stat = full.stat()
    _tiff(full, np.array([[0, 300], [100, 200]], dtype=np.uint16), mode="I;16")
    os.utime(full, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000))
    updated_path = images_db.tiff_preview_file(full, thumbnail)

    assert updated_path != preview_path
    assert not preview_path.exists()
    assert updated_path.read_bytes() != old_bytes
    assert conversion_count == 2


def test_concurrent_tiff_preview_cache_misses_only_return_complete_pngs(tmp_path):
    full = tmp_path / "concurrent.tif"
    thumbnail = tmp_path / "concurrent-thumb.png"
    _tiff(full, np.arange(256, dtype=np.uint16).reshape(16, 16), mode="I;16")
    barrier = threading.Barrier(8)

    def load_preview():
        barrier.wait()
        return images_db.tiff_preview_bytes(full, thumbnail)

    with ThreadPoolExecutor(max_workers=8) as pool:
        previews = list(pool.map(lambda _index: load_preview(), range(8)))

    assert len(set(previews)) == 1
    assert previews[0].startswith(b"\x89PNG\r\n\x1a\n")
    assert _png_pixels(previews[0])[0] == (16, 16)
    cache_files = list(tmp_path.glob(".tiff-preview-v2-*.png"))
    assert len(cache_files) == 1
    assert cache_files[0].read_bytes() == previews[0]
    assert not list(tmp_path.glob(f".{cache_files[0].name}.*"))


def test_deleting_image_removes_its_preview_cache_only(tmp_path):
    first_full = tmp_path / "first.tif"
    first_thumb = tmp_path / "first.png"
    second_full = tmp_path / "second.tif"
    second_thumb = tmp_path / "second.png"
    pixels = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    _tiff(first_full, pixels, mode="I;16")
    _tiff(second_full, pixels, mode="I;16")
    first_thumb.write_bytes(b"first thumb")
    second_thumb.write_bytes(b"second thumb")
    target_cache = images_db.tiff_preview_file(first_full, first_thumb)
    unrelated_cache = images_db.tiff_preview_file(second_full, second_thumb)
    old_variant = target_cache.with_name(f"{target_cache.stem}-old.png")
    old_variant.write_bytes(target_cache.read_bytes())

    class DeleteDB(_DB):
        def delete(self, _row):
            self.deleted = True

        def commit(self):
            self.committed = True

    asyncio.run(images_db.delete_image(1, DeleteDB([_row(1, first_full, first_thumb)])))

    assert not first_full.exists() and not first_thumb.exists()
    assert not target_cache.exists() and not old_variant.exists()
    assert unrelated_cache.exists()
