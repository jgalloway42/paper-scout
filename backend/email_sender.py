"""Format and send Gmail digest email."""

import smtplib
import ssl
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from backend.security import generate_rating_token

logger = logging.getLogger(__name__)


class EmailError(Exception):
    pass


def send_digest_email(digest_id: int, items: list[dict], recipient: str) -> None:
    """Build plain-text + HTML email and send via Gmail SMTP_SSL.

    Raises EmailError on SMTP failure.
    """
    from backend.config import get_settings

    settings = get_settings()
    sender = settings.gmail_address
    password = settings.gmail_app_password
    api_url = settings.api_base_url.rstrip("/")
    secret = settings.hmac_secret

    subject = settings.email.subject_template.format(
        date=items[0].get("published_date", "") if items else ""
    )

    plain = _build_plain(items, api_url, secret)
    html = _build_html(items, api_url, secret)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
    except smtplib.SMTPException as exc:
        raise EmailError(str(exc)) from exc


def _item_label(item: dict) -> str:
    pos = item.get("position", 0)
    if item.get("is_wildcard"):
        return f"Wildcard {pos - 2}"
    return f"Exploit Pick {pos + 1}"


def _rating_links(item: dict, api_url: str, secret: str) -> tuple[str, str]:
    item_id = item["id"]
    up_token = generate_rating_token(item_id, "up", secret)
    down_token = generate_rating_token(item_id, "down", secret)
    up_url = f"{api_url}/rate?item_id={item_id}&rating=up&token={up_token}"
    down_url = f"{api_url}/rate?item_id={item_id}&rating=down&token={down_token}"
    return up_url, down_url


def _build_plain(items: list[dict], api_url: str, secret: str) -> str:
    lines = []
    for item in items:
        up_url, down_url = _rating_links(item, api_url, secret)
        lines.append(f"[{_item_label(item)}]")
        lines.append(f"Title: {item.get('title', '')}")
        lines.append(f"Source: {item.get('source', '')} | Topic: {item.get('topic_bucket', '')}")
        lines.append(f"Published: {item.get('published_date', '')}")
        lines.append(f"\n{item.get('summary', '')}")
        lines.append(f"\nRead: {item.get('url', '')}")
        lines.append(f"👍 {up_url}")
        lines.append(f"👎 {down_url}")
        lines.append("")
    return "\n".join(lines)


def _build_html(items: list[dict], api_url: str, secret: str) -> str:
    parts = ["<html><body>"]
    for item in items:
        up_url, down_url = _rating_links(item, api_url, secret)
        parts.append(f"<h3>{_item_label(item)}: {item.get('title', '')}</h3>")
        parts.append(
            f"<p><b>Source:</b> {item.get('source', '')} &nbsp;|&nbsp; "
            f"<b>Topic:</b> {item.get('topic_bucket', '')} &nbsp;|&nbsp; "
            f"<b>Published:</b> {item.get('published_date', '')}</p>"
        )
        parts.append(f"<p>{item.get('summary', '')}</p>")
        parts.append(f'<p><a href="{item.get("url", "")}">Read paper</a></p>')
        parts.append(
            f'<p><a href="{up_url}">👍 Thumbs up</a> &nbsp; '
            f'<a href="{down_url}">👎 Thumbs down</a></p>'
        )
        parts.append("<hr>")
    parts.append("</body></html>")
    return "\n".join(parts)
