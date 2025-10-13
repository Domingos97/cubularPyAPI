from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user, get_current_admin_user, get_db
from app.models.models import User
from app.models.schemas import SuccessResponse
from app.services.email_service import EmailService
from pydantic import BaseModel, EmailStr

router = APIRouter()

class TestEmailRequest(BaseModel):
    to: EmailStr
    message: str = "This is a test email from Cubular API."

class ConfirmationEmailRequest(BaseModel):
    to: EmailStr
    username: str
    confirmation_link: str

class PasswordResetEmailRequest(BaseModel):
    to: EmailStr
    username: str
    reset_link: str

class NotificationEmailRequest(BaseModel):
    to: EmailStr
    username: str
    notification_title: str
    notification_message: str
    action_link: str = None

class SecurityAlertEmailRequest(BaseModel):
    to: EmailStr
    username: str
    alert_type: str
    alert_details: str
    ip_address: str = None
    user_agent: str = None

@router.get("/test-connection", response_model=Dict[str, Any], tags=["email"])
async def test_smtp_connection(
    current_user: User = Depends(get_current_admin_user)
):
    """
    Test SMTP connection and configuration (admin only)
    """
    try:
        result = EmailService.test_smtp_connection()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test SMTP connection: {str(e)}"
        )

@router.post("/send-test", response_model=SuccessResponse, tags=["email"])
async def send_test_email(
    email_data: TestEmailRequest,
    current_user: User = Depends(get_current_admin_user)
):
    """
    Send a test email (admin only)
    """
    try:
        success = await EmailService.send_email(
            to=str(email_data.to),
            subject="Test Email from Cubular",
            html_content=f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <h2>Test Email</h2>
                <p>This is a test email from the Cubular API.</p>
                <p>Message: {email_data.message}</p>
                <p>Sent by: {current_user.username} ({current_user.email})</p>
                <p>If you received this email, the email service is working correctly.</p>
            </body>
            </html>
            """
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send test email"
            )
        
        return SuccessResponse(message="Test email sent successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send test email: {str(e)}"
        )

@router.post("/send-confirmation", response_model=SuccessResponse, tags=["email"])
async def send_confirmation_email(
    email_data: ConfirmationEmailRequest,
    current_user: User = Depends(get_current_admin_user)
):
    """
    Send email confirmation email (admin only)
    """
    try:
        success = await EmailService.send_confirmation_email(
            to=str(email_data.to),
            username=email_data.username,
            confirmation_link=email_data.confirmation_link
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send confirmation email"
            )
        
        return SuccessResponse(message="Confirmation email sent successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send confirmation email: {str(e)}"
        )

@router.post("/send-password-reset", response_model=SuccessResponse, tags=["email"])
async def send_password_reset_email(
    email_data: PasswordResetEmailRequest,
    current_user: User = Depends(get_current_admin_user)
):
    """
    Send password reset email (admin only)
    """
    try:
        success = await EmailService.send_password_reset_email(
            to=str(email_data.to),
            username=email_data.username,
            reset_link=email_data.reset_link
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send password reset email"
            )
        
        return SuccessResponse(message="Password reset email sent successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send password reset email: {str(e)}"
        )

@router.post("/send-welcome", response_model=SuccessResponse, tags=["email"])
async def send_welcome_email(
    to: EmailStr,
    username: str,
    current_user: User = Depends(get_current_admin_user)
):
    """
    Send welcome email (admin only)
    """
    try:
        success = await EmailService.send_welcome_email(
            to=str(to),
            username=username
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send welcome email"
            )
        
        return SuccessResponse(message="Welcome email sent successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send welcome email: {str(e)}"
        )

@router.post("/send-notification", response_model=SuccessResponse, tags=["email"])
async def send_notification_email(
    email_data: NotificationEmailRequest,
    current_user: User = Depends(get_current_admin_user)
):
    """
    Send notification email (admin only)
    """
    try:
        success = await EmailService.send_notification_email(
            to=str(email_data.to),
            username=email_data.username,
            notification_title=email_data.notification_title,
            notification_message=email_data.notification_message,
            action_link=email_data.action_link
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send notification email"
            )
        
        return SuccessResponse(message="Notification email sent successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send notification email: {str(e)}"
        )

@router.post("/send-security-alert", response_model=SuccessResponse, tags=["email"])
async def send_security_alert_email(
    email_data: SecurityAlertEmailRequest,
    current_user: User = Depends(get_current_admin_user)
):
    """
    Send security alert email (admin only)
    """
    try:
        success = await EmailService.send_security_alert_email(
            to=str(email_data.to),
            username=email_data.username,
            alert_type=email_data.alert_type,
            alert_details=email_data.alert_details,
            ip_address=email_data.ip_address,
            user_agent=email_data.user_agent
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send security alert email"
            )
        
        return SuccessResponse(message="Security alert email sent successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send security alert email: {str(e)}"
        )