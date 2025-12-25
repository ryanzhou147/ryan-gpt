import re

def mask_email(text: str, pattern: str = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}") -> tuple[str, int]:

    masked_text, num_subs = re.subn(pattern, "|||EMAIL_ADDRESS|||", text)
    return masked_text, num_subs

def mask_phone_number(text: str, pattern: str = r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}") -> tuple[str, int]:
    masked_text, num_subs = re.subn(pattern, "|||PHONE_NUMBER|||", text)
    return masked_text, num_subs