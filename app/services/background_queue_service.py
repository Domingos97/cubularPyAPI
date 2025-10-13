"""
Background Queue Service
Simple background processing queue for non-critical operations
Ported from TypeScript backgroundQueue.ts
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from app.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class QueueJob:
    """Queue job data structure"""
    id: str
    type: str
    payload: Any
    retries: int
    max_retries: int
    created_at: datetime

class BackgroundQueue:
    """
    Simple Background Processing Queue
    Processes non-critical operations asynchronously to improve user experience
    """
    
    _instance: Optional['BackgroundQueue'] = None
    
    def __new__(cls) -> 'BackgroundQueue':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.queue: List[QueueJob] = []
        self.processing = False
        self.MAX_RETRIES = 3
        self.PROCESS_INTERVAL = 1  # 1 second
        self._initialized = True
        self._task = None
        
        # Don't start processing queue immediately - wait for event loop
    
    def ensure_processing_started(self):
        """Ensure background processing is started if event loop is available"""
        try:
            if self._task is None or self._task.done():
                loop = asyncio.get_running_loop()
                self._task = loop.create_task(self._start_processing())
        except RuntimeError:
            # No event loop running, will be started later
            pass
    
    async def _start_processing(self):
        """Start processing the queue"""
        try:
            while True:
                if not self.processing and self.queue:
                    await self._process_queue()
                await asyncio.sleep(self.PROCESS_INTERVAL)
        except Exception as e:
            logger.error(f"Background queue processing error: {e}")
    
    async def _process_queue(self):
        """Process jobs in the queue"""
        if self.processing or not self.queue:
            return
        
        self.processing = True
        
        try:
            # Process jobs one by one
            while self.queue:
                job = self.queue.pop(0)
                
                try:
                    await self._process_job(job)
                    logger.info(f"Successfully processed job {job.id} of type {job.type}")
                except Exception as e:
                    logger.error(f"Error processing job {job.id}: {e}")
                    
                    # Retry logic
                    if job.retries < job.max_retries:
                        job.retries += 1
                        self.queue.append(job)
                        logger.info(f"Retrying job {job.id} (attempt {job.retries}/{job.max_retries})")
                    else:
                        logger.error(f"Job {job.id} failed after {job.max_retries} retries")
                
        finally:
            self.processing = False
    
    async def _process_job(self, job: QueueJob):
        """Process a single job"""
        # Simulate job processing based on type
        if job.type == 'survey_analysis':
            await self._process_survey_analysis(job.payload)
        elif job.type == 'file_processing':
            await self._process_file_processing(job.payload)
        elif job.type == 'email_notification':
            await self._process_email_notification(job.payload)
        elif job.type == 'data_cleanup':
            await self._process_data_cleanup(job.payload)
        else:
            logger.warning(f"Unknown job type: {job.type}")
    
    async def _process_survey_analysis(self, payload: Dict[str, Any]):
        """Process survey analysis job"""
        survey_id = payload.get('survey_id')
        logger.info(f"Processing survey analysis for survey {survey_id}")
        # Simulate processing time
        await asyncio.sleep(0.1)
    
    async def _process_file_processing(self, payload: Dict[str, Any]):
        """Process file processing job"""
        file_id = payload.get('file_id')
        logger.info(f"Processing file {file_id}")
        # Simulate processing time
        await asyncio.sleep(0.1)
    
    async def _process_email_notification(self, payload: Dict[str, Any]):
        """Process email notification job"""
        recipient = payload.get('recipient')
        logger.info(f"Sending email to {recipient}")
        # Simulate processing time
        await asyncio.sleep(0.1)
    
    async def _process_data_cleanup(self, payload: Dict[str, Any]):
        """Process data cleanup job"""
        cleanup_type = payload.get('type')
        logger.info(f"Performing data cleanup: {cleanup_type}")
        # Simulate processing time
        await asyncio.sleep(0.1)
    
    def add_job(self, job_type: str, payload: Any, max_retries: Optional[int] = None) -> str:
        """Add a job to the queue"""
        job_id = str(uuid.uuid4())
        
        job = QueueJob(
            id=job_id,
            type=job_type,
            payload=payload,
            retries=0,
            max_retries=max_retries or self.MAX_RETRIES,
            created_at=datetime.now()
        )
        
        self.queue.append(job)
        logger.info(f"Added job {job_id} of type {job_type} to queue")
        
        # Ensure processing is started
        self.ensure_processing_started()
        
        return job_id
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "queue_length": len(self.queue),
            "processing": self.processing,
            "total_jobs_by_type": self._get_jobs_by_type()
        }
    
    def _get_jobs_by_type(self) -> Dict[str, int]:
        """Get count of jobs by type"""
        jobs_by_type: Dict[str, int] = {}
        for job in self.queue:
            jobs_by_type[job.type] = jobs_by_type.get(job.type, 0) + 1
        return jobs_by_type
    
    def clear_queue(self):
        """Clear all jobs from queue"""
        self.queue.clear()
        logger.info("Queue cleared")

# Singleton instance
background_queue = BackgroundQueue()

# Utility functions for easy access
async def add_survey_analysis_job(survey_id: str) -> str:
    """Add survey analysis job to queue"""
    return background_queue.add_job('survey_analysis', {'survey_id': survey_id})

async def add_file_processing_job(file_id: str, file_path: str) -> str:
    """Add file processing job to queue"""
    return background_queue.add_job('file_processing', {'file_id': file_id, 'file_path': file_path})

async def add_email_notification_job(recipient: str, subject: str, body: str) -> str:
    """Add email notification job to queue"""
    return background_queue.add_job('email_notification', {
        'recipient': recipient,
        'subject': subject,
        'body': body
    })

async def add_data_cleanup_job(cleanup_type: str, parameters: Dict[str, Any]) -> str:
    """Add data cleanup job to queue"""
    return background_queue.add_job('data_cleanup', {
        'type': cleanup_type,
        'parameters': parameters
    })

def get_queue_status() -> Dict[str, Any]:
    """Get current queue status"""
    return background_queue.get_queue_status()