from io import BytesIO

import pytest
from PIL import Image

from pkrvision.image_io import ImageDecodeError, decode_image


def image_bytes(format_name: str = "PNG", size: tuple[int, int] = (12, 8)) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, "white").save(stream, format=format_name)
    return stream.getvalue()


def test_decode_supported_image() -> None:
    decoded = decode_image(image_bytes(), max_pixels=200)
    assert decoded.image.size == (12, 8)
    assert decoded.array.shape == (8, 12, 3)


@pytest.mark.parametrize("payload", [b"", b"not an image"])
def test_decode_rejects_invalid_payload(payload: bytes) -> None:
    with pytest.raises(ImageDecodeError):
        decode_image(payload, max_pixels=200)


def test_decode_enforces_pixel_budget() -> None:
    with pytest.raises(ImageDecodeError, match="pixel limit"):
        decode_image(image_bytes(size=(20, 20)), max_pixels=399)
