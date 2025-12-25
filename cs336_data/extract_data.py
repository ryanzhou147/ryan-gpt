from fastwarc.warc import ArchiveIterator, WarcRecordType

# Write a function that extracts text from a byte string containing raw HTML. Use
# resiliparse.extract.html2text.extract_plain_text to perform the extraction. This function needs a string, so you will need to first decode the byte string into a Unicode string. Be
# aware that the input byte string might not be encoded in UTF-8, so your function should be able
# to detect the encoding in case UTF-8 fails. Resiliparse also offers
# resiliparse.parse.encoding.detect_encoding(), which might be useful.
# Deliverable: A function that takes a byte string containing HTML and returns a string containing the extracted text. Implement the adapter [run_extract_text_from_html_bytes] and
# make sure it passes uv run pytest -k test_extract_text_from_html_bytes

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str:
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.encoding import detect_encoding

    try:
        html_str = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding)
    return extract_plain_text(html_str)