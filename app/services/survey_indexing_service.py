"""
Survey Indexing Service
Generates embeddings for survey responses and creates pickle files
for use with the fast_search_service
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SurveyIndexingService:
    """
    Service to generate embeddings and pickle files for survey data
    Creates files in the format expected by fast_search_service
    """
    
    def __init__(self):
        self.embedding_service = embedding_service
    
    async def index_survey_file(
        self, 
        survey_id: str, 
        file_id: str, 
        processed_data: Dict[str, Any],
        db: AsyncSession = None
    ) -> bool:
        """
        Generate embeddings for survey responses and create pickle file
        
        Args:
            survey_id: UUID of the survey
            file_id: UUID of the file
            processed_data: Processed survey data from FileProcessor
            db: Database session for module configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Starting indexing for survey {survey_id}, file {file_id}")
            
            # Extract responses from processed data
            responses = processed_data.get('responses', [])
            if not responses:
                logger.warning(f"No responses found in processed data for survey {survey_id}, file {file_id}")
                return False
            
            # Extract text content for embedding generation
            texts = []
            semantic_dict = []
            
            for i, response in enumerate(responses):
                text = response.get('text', '').strip()
                if text:
                    texts.append(text)
                    
                    # Create semantic_dict entry expected by fast_search_service
                    semantic_entry = {
                        'index': i,
                        'cleanedText': text,
                        'originalText': text,
                        'demographics': response.get('demographics', {}),
                        'metadata': response.get('raw_data', {})
                    }
                    semantic_dict.append(semantic_entry)
            
            if not texts:
                logger.warning(f"No valid text content found for embedding generation in survey {survey_id}, file {file_id}")
                return False
            
            logger.info(f"Generating embeddings for {len(texts)} responses...")
            
            # Generate embeddings using the configured model
            embeddings = await self.embedding_service.generate_embeddings(texts, db)
            
            if not embeddings or len(embeddings) != len(texts):
                logger.error(f"Failed to generate embeddings or embedding count mismatch for survey {survey_id}, file {file_id}")
                return False
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            
            # Prepare data structure for pickle file
            pickle_data = {
                'embeddings': embeddings,
                'texts': texts,
                'semantic_dict': semantic_dict,
                'metadata': {
                    'survey_id': survey_id,
                    'file_id': file_id,
                    'total_responses': len(texts),
                    'embedding_dimension': len(embeddings[0]) if embeddings else 0,
                    'processed_at': processed_data.get('summary', {}),
                    'columns': processed_data.get('columns', []),
                    'text_columns': processed_data.get('text_columns', []),
                    'demographic_columns': processed_data.get('demographic_columns', [])
                }
            }
            
            # Create file path: survey_data/{survey_id}/{file_id}/survey_data.pkl
            file_dir = Path(settings.upload_dir) / survey_id / file_id
            pickle_path = file_dir / "survey_data.pkl"
            
            # Ensure directory exists
            file_dir.mkdir(parents=True, exist_ok=True)
            
            # Save pickle file
            logger.info(f"Saving pickle file to: {pickle_path}")
            with open(pickle_path, 'wb') as f:
                pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Verify file was created and is readable
            if pickle_path.exists():
                # Test loading the pickle file
                with open(pickle_path, 'rb') as f:
                    test_data = pickle.load(f)
                
                if 'embeddings' in test_data and 'semantic_dict' in test_data:
                    logger.info(f"Successfully created and verified pickle file for survey {survey_id}, file {file_id}")
                    logger.info(f"Pickle file contains {len(test_data['embeddings'])} embeddings and {len(test_data['semantic_dict'])} responses")
                    return True
                else:
                    logger.error(f"Pickle file verification failed - missing required keys")
                    return False
            else:
                logger.error(f"Pickle file was not created at {pickle_path}")
                return False
            
        except Exception as e:
            logger.error(f"Error indexing survey file {survey_id}/{file_id}: {str(e)}", exc_info=True)
            return False
    
    async def index_multiple_files(
        self, 
        survey_id: str, 
        file_data: List[Dict[str, Any]],
        db: AsyncSession = None
    ) -> Dict[str, bool]:
        """
        Index multiple files for a survey
        
        Args:
            survey_id: UUID of the survey
            file_data: List of dicts with 'file_id' and 'processed_data'
            db: Database session
            
        Returns:
            Dict mapping file_id to success status
        """
        results = {}
        
        for file_info in file_data:
            file_id = file_info.get('file_id')
            processed_data = file_info.get('processed_data')
            
            if not file_id or not processed_data:
                logger.warning(f"Skipping invalid file info: {file_info}")
                results[file_id or 'unknown'] = False
                continue
            
            success = await self.index_survey_file(survey_id, file_id, processed_data, db)
            results[file_id] = success
        
        return results
    
    def get_pickle_file_path(self, survey_id: str, file_id: str) -> Path:
        """Get the path where a pickle file should be located"""
        return Path(settings.upload_dir) / survey_id / file_id / "survey_data.pkl"
    
    def pickle_file_exists(self, survey_id: str, file_id: str) -> bool:
        """Check if pickle file exists for a survey file"""
        pickle_path = self.get_pickle_file_path(survey_id, file_id)
        return pickle_path.exists()
    
    async def reindex_survey_file(
        self, 
        survey_id: str, 
        file_id: str,
        db: AsyncSession = None
    ) -> bool:
        """
        Reindex an existing survey file by re-reading the original file
        and regenerating embeddings
        """
        try:
            # Find the original file
            file_dir = Path(settings.upload_dir) / survey_id / file_id
            
            # Look for original file
            original_file = None
            for filename in ['original_file.csv', 'original_file.xlsx']:
                potential_file = file_dir / filename
                if potential_file.exists():
                    original_file = potential_file
                    break
            
            if not original_file:
                logger.error(f"No original file found for survey {survey_id}, file {file_id}")
                return False
            
            logger.info(f"Reindexing from original file: {original_file}")
            
            # Import FileProcessor to reprocess the file
            from app.services.survey_service import FileProcessor
            
            # Reprocess the file
            processed_data = await FileProcessor.process_survey_file(
                str(original_file), 
                original_file.name
            )
            
            # Generate new embeddings and pickle file
            return await self.index_survey_file(survey_id, file_id, processed_data, db)
            
        except Exception as e:
            logger.error(f"Error reindexing survey file {survey_id}/{file_id}: {str(e)}")
            return False


# Create service instance
survey_indexing_service = SurveyIndexingService()