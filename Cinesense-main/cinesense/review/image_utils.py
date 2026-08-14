"""Optimize Review.photo uploads with Pillow (resize + light recompression)."""

from __future__ import annotations

import os
from io import BytesIO

from django.core.files.base import ContentFile
from PIL import Image, ImageOps

# List cards are ~18rem (~288px); analysis thumbs are ~120px. 1200px on the
# longest edge covers high-DPI displays with headroom without storing camera originals.
PHOTO_MAX_EDGE = 1200
JPEG_QUALITY = 85
WEBP_QUALITY = 85


def _format_and_extension(image: Image.Image, upload_name: str) -> tuple[str, str]:
    fmt = (image.format or '').upper()
    if fmt == 'JPG':
        fmt = 'JPEG'
    if fmt not in {'JPEG', 'PNG', 'WEBP', 'GIF'}:
        # Fall back from extension when Pillow did not record a format.
        ext = os.path.splitext(upload_name)[1].lower()
        fmt = {
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.png': 'PNG',
            '.webp': 'WEBP',
            '.gif': 'GIF',
        }.get(ext, 'JPEG')
    extension = {
        'JPEG': '.jpg',
        'PNG': '.png',
        'WEBP': '.webp',
        'GIF': '.gif',
    }[fmt]
    return fmt, extension


def optimize_review_photo(photo_file) -> ContentFile | None:
    """Return an optimized ContentFile, or None if no rewrite is needed.

    Skips work when the image is already within PHOTO_MAX_EDGE (after EXIF
    orientation). Preserves PNG transparency and animated GIFs. On any
    processing failure, returns None so the original upload is stored unchanged.
    """
    if not photo_file:
        return None

    original_name = getattr(photo_file, 'name', 'photo.jpg') or 'photo.jpg'

    try:
        if hasattr(photo_file, 'open'):
            try:
                photo_file.open('rb')
            except Exception:
                pass
        if hasattr(photo_file, 'seek'):
            photo_file.seek(0)
        raw = photo_file.read()
        if hasattr(photo_file, 'seek'):
            photo_file.seek(0)
        if not raw:
            return None

        image = Image.open(BytesIO(raw))

        # Animated GIFs: leave untouched to avoid dropping frames.
        if getattr(image, 'is_animated', False) and getattr(image, 'n_frames', 1) > 1:
            return None

        # Prefer header size before forcing a full decode (some tiny test PNGs
        # are accepted by Django but fail on Image.load).
        orientation = image.getexif().get(274, 1) or 1
        needs_orient = orientation != 1
        needs_resize = max(image.size) > PHOTO_MAX_EDGE
        if not needs_orient and not needs_resize:
            return None

        image = ImageOps.exif_transpose(image)
        if max(image.size) > PHOTO_MAX_EDGE:
            image.thumbnail(
                (PHOTO_MAX_EDGE, PHOTO_MAX_EDGE),
                Image.Resampling.LANCZOS,
            )

        fmt, extension = _format_and_extension(image, original_name)

        buffer = BytesIO()
        save_kwargs: dict = {}

        if fmt == 'JPEG':
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                alpha = image.split()[-1]
                background.paste(image, mask=alpha)
                image = background
            elif image.mode == 'P':
                image = image.convert('RGBA')
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            save_kwargs = {'quality': JPEG_QUALITY, 'optimize': True}
        elif fmt == 'PNG':
            if image.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P'):
                image = image.convert('RGBA')
            save_kwargs = {'optimize': True}
        elif fmt == 'WEBP':
            save_kwargs = {'quality': WEBP_QUALITY, 'method': 4}
        elif fmt == 'GIF':
            if image.mode not in ('P', 'L', 'RGB'):
                image = image.convert('P')
            save_kwargs = {'optimize': True}

        image.save(buffer, format=fmt, **save_kwargs)
        buffer.seek(0)

        base = os.path.splitext(os.path.basename(original_name))[0] or 'photo'
        return ContentFile(buffer.getvalue(), name=f'{base}{extension}')
    except Exception:
        if hasattr(photo_file, 'seek'):
            try:
                photo_file.seek(0)
            except Exception:
                pass
        return None
