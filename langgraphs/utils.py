from pathlib import Path
import base64

# Directory where test images live (relative to this utils file)
HERE = Path(__file__).parent
TEST_IMAGES_DIR = HERE / "test_images"

def encode_image_to_data_url(path: str) -> str:
    """
    If `path` is already a data‑URL or HTTP(S) URL, return it unchanged.
    Otherwise, attempt to locate the file on disk:
      1) As given (absolute or relative to cwd)
      2) Under langgraphs/test_images/
    If found, read & base64‑encode it into a data‑URL.
    Otherwise, return the original path (will error if not a valid URL).
    """
    # Pass through URLs and data‑URLs
    if path.startswith(("data:", "http://", "https://")):
        return path

    # Try the raw path first
    file_path = Path(path)
    if not file_path.is_file():
        # Fallback: look under the test_images folder
        file_path = TEST_IMAGES_DIR / path

    if file_path.is_file():
        # Infer MIME type from extension
        ext = file_path.suffix.lower().lstrip(".")
        mime = "image/png" if ext == "png" else f"image/{ext}"
        data = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{data}"

    # Last resort: return as‑is
    return path

