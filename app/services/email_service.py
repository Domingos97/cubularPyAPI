from typing import Optional, Dict, Any
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import aiofiles
from pathlib import Path
from datetime import datetime

from app.core.config import settings


class EmailService:
    """Email service for sending various types of emails"""
    
    @staticmethod
    def _create_smtp_connection():
        """Create and configure SMTP connection"""
        if not all([settings.SMTP_HOST, settings.SMTP_PORT, settings.SMTP_USER, settings.SMTP_PASS]):
            raise ValueError("SMTP configuration is incomplete. Please check environment variables.")
        
        # Create secure SSL context
        context = ssl.create_default_context()
        
        # Connect to server
        if settings.SMTP_TLS:
            # Use TLS
            server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT)
            server.starttls(context=context)
        else:
            # Use SSL
            server = smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, context=context)
        
        # Login
        server.login(settings.SMTP_USER, settings.SMTP_PASS)
        
        return server
    
    @staticmethod
    async def _load_email_template(template_name: str, variables: Dict[str, Any]) -> str:
        """Load and process email template"""
        template_path = Path(__file__).parent.parent / "templates" / f"{template_name}.html"
        
        if not template_path.exists():
            # Return basic template if file doesn't exist
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Cubular Notification</title>
                <style>
                    body {{ font-family: Arial, sans-serif; background: #f6f8fa; color: #222; }}
                    .container {{ max-width: 480px; margin: 40px auto; background: #fff; 
                                border-radius: 8px; box-shadow: 0 2px 8px #eee; padding: 32px; }}
                    .btn {{ display: inline-block; background: #3b82f6; color: #fff; 
                           padding: 12px 24px; border-radius: 4px; text-decoration: none; font-weight: bold; }}
                    .footer {{ color: #888; font-size: 12px; margin-top: 32px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Cubular Notification</h2>
                    <p>Hi {variables.get('username', 'User')},</p>
                    <p>{variables.get('message', 'Thank you for using Cubular.')}</p>
                    <div class="footer">
                        &copy; {datetime.now().year} Cubular. All rights reserved.
                    </div>
                </div>
            </body>
            </html>
            """
        
        try:
            async with aiofiles.open(template_path, 'r', encoding='utf-8') as f:
                template = await f.read()
            
            # Replace variables
            for key, value in variables.items():
                template = template.replace(f"{{{{{key}}}}}", str(value))
            
            # Replace year if not provided
            if "{{year}}" in template:
                template = template.replace("{{year}}", str(datetime.now().year))
            
            return template
            
        except Exception as e:
            raise ValueError(f"Failed to load email template {template_name}: {str(e)}")
    
    @staticmethod
    async def send_email(
        to: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
        from_email: Optional[str] = None,
        attachments: Optional[list] = None
    ) -> bool:
        """Send email with HTML content"""
        try:
            # Create message container
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = from_email or settings.SMTP_FROM or settings.SMTP_USER
            msg['To'] = to
            
            # Add text part if provided
            if text_content:
                text_part = MIMEText(text_content, 'plain')
                msg.attach(text_part)
            
            # Add HTML part
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Add attachments if provided
            if attachments:
                for attachment in attachments:
                    if os.path.isfile(attachment):
                        with open(attachment, "rb") as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                        
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {os.path.basename(attachment)}'
                        )
                        msg.attach(part)
            
            # Send email
            server = EmailService._create_smtp_connection()
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            return False
    
    @staticmethod
    async def send_confirmation_email(
        to: str,
        username: str,
        confirmation_link: str
    ) -> bool:
        """Send email confirmation email"""
        try:
            variables = {
                'username': username,
                'confirmation_link': confirmation_link,
                'year': datetime.now().year
            }
            
            html_content = await EmailService._load_email_template('confirmation_email', variables)
            
            return await EmailService.send_email(
                to=to,
                subject="Confirm your email for Cubular",
                html_content=html_content
            )
            
        except Exception as e:
            print(f"Failed to send confirmation email: {str(e)}")
            return False
    
    @staticmethod
    async def send_password_reset_email(
        to: str,
        username: str,
        reset_link: str
    ) -> bool:
        """Send password reset email"""
        try:
            variables = {
                'username': username,
                'reset_link': reset_link,
                'year': datetime.now().year
            }
            
            html_content = await EmailService._load_email_template('password_reset_email', variables)
            
            return await EmailService.send_email(
                to=to,
                subject="Reset your Cubular password",
                html_content=html_content
            )
            
        except Exception as e:
            print(f"Failed to send password reset email: {str(e)}")
            return False
    
    @staticmethod
    async def send_welcome_email(
        to: str,
        username: str
    ) -> bool:
        """Send welcome email after email confirmation"""
        try:
            variables = {
                'username': username,
                'year': datetime.now().year
            }
            
            html_content = await EmailService._load_email_template('welcome_email', variables)
            
            return await EmailService.send_email(
                to=to,
                subject="Welcome to Cubular!",
                html_content=html_content
            )
            
        except Exception as e:
            print(f"Failed to send welcome email: {str(e)}")
            return False
    
    @staticmethod
    async def send_notification_email(
        to: str,
        username: str,
        notification_title: str,
        notification_message: str,
        action_link: Optional[str] = None
    ) -> bool:
        """Send general notification email"""
        try:
            variables = {
                'username': username,
                'notification_title': notification_title,
                'notification_message': notification_message,
                'action_link': action_link or '',
                'year': datetime.now().year
            }
            
            html_content = await EmailService._load_email_template('notification_email', variables)
            
            return await EmailService.send_email(
                to=to,
                subject=f"Cubular: {notification_title}",
                html_content=html_content
            )
            
        except Exception as e:
            print(f"Failed to send notification email: {str(e)}")
            return False
    
    @staticmethod
    async def send_security_alert_email(
        to: str,
        username: str,
        alert_type: str,
        alert_details: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Send security alert email"""
        try:
            variables = {
                'username': username,
                'alert_type': alert_type,
                'alert_details': alert_details,
                'ip_address': ip_address or 'Unknown',
                'user_agent': user_agent or 'Unknown',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'year': datetime.now().year
            }
            
            html_content = await EmailService._load_email_template('security_alert_email', variables)
            
            return await EmailService.send_email(
                to=to,
                subject=f"Cubular Security Alert: {alert_type}",
                html_content=html_content
            )
            
        except Exception as e:
            print(f"Failed to send security alert email: {str(e)}")
            return False
    
    @staticmethod
    def test_smtp_connection() -> Dict[str, Any]:
        """Test SMTP connection and configuration"""
        try:
            server = EmailService._create_smtp_connection()
            server.quit()
            
            return {
                "status": "success",
                "message": "SMTP connection successful",
                "config": {
                    "host": settings.SMTP_HOST,
                    "port": settings.SMTP_PORT,
                    "user": settings.SMTP_USER,
                    "tls": settings.SMTP_TLS,
                    "from": settings.SMTP_FROM or settings.SMTP_USER
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"SMTP connection failed: {str(e)}",
                "config": {
                    "host": settings.SMTP_HOST,
                    "port": settings.SMTP_PORT,
                    "user": settings.SMTP_USER,
                    "tls": settings.SMTP_TLS
                }
            }


# Create singleton instance
email_service = EmailService()