from email.mime.text import MIMEText
import base64
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/gmail.send",
]


def get_services():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES,
            )
            creds = flow.run_local_server(
                host="localhost",
                port=8080,
                open_browser=False,
            )

        with open("token.json", "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    drive_service = build("drive", "v3", credentials=creds)
    gmail_service = build("gmail", "v1", credentials=creds)

    return drive_service, gmail_service


def upload_file_to_drive(drive_service, file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    file_metadata = {
        "name": file_path.name,
    }

    media = MediaFileUpload(
        str(file_path),
        mimetype="application/octet-stream",
        resumable=True,
    )

    uploaded_file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, webViewLink",
    ).execute()

    file_id = uploaded_file["id"]

    drive_service.permissions().create(
        fileId=file_id,
        body={
            "type": "anyone",
            "role": "reader",
        },
    ).execute()

    file_info = drive_service.files().get(
        fileId=file_id,
        fields="webViewLink",
    ).execute()

    return file_info["webViewLink"]


def send_email(gmail_service, to_email, subject, body):
    message = MIMEText(body, "plain", "utf-8")
    message["to"] = to_email
    message["subject"] = subject

    raw_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode("utf-8")

    gmail_service.users().messages().send(
        userId="me",
        body={"raw": raw_message},
    ).execute()


def send_result(pt_path, to_email="enjospoti@gmail.com"):
    drive_service, gmail_service = get_services()

    link = upload_file_to_drive(drive_service, pt_path)

    send_email(
        gmail_service,
        to_email,
        "PT 파일 공유 링크입니다",
        f"아래 링크에서 .pt 파일을 다운로드할 수 있습니다.\n\n{link}",
    )

    print("업로드 및 이메일 발송 완료")
    print(link)