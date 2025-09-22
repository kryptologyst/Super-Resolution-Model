import sqlite3
import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ImageRecord:
    """Data class for image records"""
    id: Optional[int] = None
    filename: str = ""
    original_path: str = ""
    processed_path: str = ""
    model_name: str = ""
    scale_factor: int = 4
    processing_time: float = 0.0
    file_size_original: int = 0
    file_size_processed: int = 0
    image_width: int = 0
    image_height: int = 0
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DatabaseManager:
    """Manages SQLite database for storing image processing records"""
    
    def __init__(self, db_path: str = "database/images.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create images table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        original_path TEXT NOT NULL,
                        processed_path TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        scale_factor INTEGER NOT NULL,
                        processing_time REAL NOT NULL,
                        file_size_original INTEGER NOT NULL,
                        file_size_processed INTEGER NOT NULL,
                        image_width INTEGER NOT NULL,
                        image_height INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT
                    )
                """)
                
                # Create processing_history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processing_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id INTEGER NOT NULL,
                        model_name TEXT NOT NULL,
                        scale_factor INTEGER NOT NULL,
                        processing_time REAL NOT NULL,
                        metrics TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (image_id) REFERENCES images (id)
                    )
                """)
                
                # Create model_performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        avg_processing_time REAL NOT NULL,
                        avg_psnr REAL,
                        avg_ssim REAL,
                        avg_ms_ssim REAL,
                        avg_lpips REAL,
                        total_images INTEGER NOT NULL,
                        last_updated TEXT NOT NULL
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def add_image_record(self, record: ImageRecord) -> int:
        """Add a new image record to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO images (
                        filename, original_path, processed_path, model_name,
                        scale_factor, processing_time, file_size_original,
                        file_size_processed, image_width, image_height,
                        created_at, updated_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.filename,
                    record.original_path,
                    record.processed_path,
                    record.model_name,
                    record.scale_factor,
                    record.processing_time,
                    record.file_size_original,
                    record.file_size_processed,
                    record.image_width,
                    record.image_height,
                    record.created_at,
                    record.updated_at,
                    json.dumps(record.metadata)
                ))
                
                record_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Added image record with ID: {record_id}")
                return record_id
                
        except Exception as e:
            logger.error(f"Failed to add image record: {e}")
            raise
    
    def get_image_record(self, record_id: int) -> Optional[ImageRecord]:
        """Get an image record by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM images WHERE id = ?
                """, (record_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_image_record(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get image record: {e}")
            return None
    
    def get_all_image_records(self, limit: int = 100) -> List[ImageRecord]:
        """Get all image records with optional limit"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM images ORDER BY created_at DESC LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                return [self._row_to_image_record(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get image records: {e}")
            return []
    
    def search_image_records(self, model_name: Optional[str] = None,
                           scale_factor: Optional[int] = None,
                           date_from: Optional[str] = None,
                           date_to: Optional[str] = None) -> List[ImageRecord]:
        """Search image records with filters"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM images WHERE 1=1"
                params = []
                
                if model_name:
                    query += " AND model_name = ?"
                    params.append(model_name)
                
                if scale_factor:
                    query += " AND scale_factor = ?"
                    params.append(scale_factor)
                
                if date_from:
                    query += " AND created_at >= ?"
                    params.append(date_from)
                
                if date_to:
                    query += " AND created_at <= ?"
                    params.append(date_to)
                
                query += " ORDER BY created_at DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [self._row_to_image_record(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to search image records: {e}")
            return []
    
    def update_image_record(self, record_id: int, updates: Dict[str, Any]) -> bool:
        """Update an image record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build update query
                set_clauses = []
                params = []
                
                for key, value in updates.items():
                    if key == 'metadata':
                        value = json.dumps(value)
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
                
                params.append(record_id)
                
                query = f"UPDATE images SET {', '.join(set_clauses)}, updated_at = ? WHERE id = ?"
                params.append(datetime.now().isoformat())
                
                cursor.execute(query, params)
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Updated image record {record_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to update image record: {e}")
            return False
    
    def delete_image_record(self, record_id: int) -> bool:
        """Delete an image record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM images WHERE id = ?", (record_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Deleted image record {record_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete image record: {e}")
            return False
    
    def add_processing_history(self, image_id: int, model_name: str,
                              scale_factor: int, processing_time: float,
                              metrics: Optional[Dict[str, float]] = None) -> int:
        """Add processing history record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO processing_history (
                        image_id, model_name, scale_factor, processing_time,
                        metrics, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    image_id,
                    model_name,
                    scale_factor,
                    processing_time,
                    json.dumps(metrics) if metrics else None,
                    datetime.now().isoformat()
                ))
                
                history_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Added processing history with ID: {history_id}")
                return history_id
                
        except Exception as e:
            logger.error(f"Failed to add processing history: {e}")
            raise
    
    def get_model_performance(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get performance statistics for a model"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM model_performance WHERE model_name = ?
                """, (model_name,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'model_name': row[1],
                        'avg_processing_time': row[2],
                        'avg_psnr': row[3],
                        'avg_ssim': row[4],
                        'avg_ms_ssim': row[5],
                        'avg_lpips': row[6],
                        'total_images': row[7],
                        'last_updated': row[8]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            return None
    
    def update_model_performance(self, model_name: str, performance_data: Dict[str, Any]) -> bool:
        """Update model performance statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if record exists
                cursor.execute("SELECT id FROM model_performance WHERE model_name = ?", (model_name,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    cursor.execute("""
                        UPDATE model_performance SET
                            avg_processing_time = ?, avg_psnr = ?, avg_ssim = ?,
                            avg_ms_ssim = ?, avg_lpips = ?, total_images = ?,
                            last_updated = ?
                        WHERE model_name = ?
                    """, (
                        performance_data.get('avg_processing_time', 0),
                        performance_data.get('avg_psnr', 0),
                        performance_data.get('avg_ssim', 0),
                        performance_data.get('avg_ms_ssim', 0),
                        performance_data.get('avg_lpips', 0),
                        performance_data.get('total_images', 0),
                        datetime.now().isoformat(),
                        model_name
                    ))
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO model_performance (
                            model_name, avg_processing_time, avg_psnr, avg_ssim,
                            avg_ms_ssim, avg_lpips, total_images, last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        model_name,
                        performance_data.get('avg_processing_time', 0),
                        performance_data.get('avg_psnr', 0),
                        performance_data.get('avg_ssim', 0),
                        performance_data.get('avg_ms_ssim', 0),
                        performance_data.get('avg_lpips', 0),
                        performance_data.get('total_images', 0),
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
                logger.info(f"Updated model performance for {model_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update model performance: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total images
                cursor.execute("SELECT COUNT(*) FROM images")
                total_images = cursor.fetchone()[0]
                
                # Get total processing time
                cursor.execute("SELECT SUM(processing_time) FROM images")
                total_processing_time = cursor.fetchone()[0] or 0
                
                # Get model distribution
                cursor.execute("""
                    SELECT model_name, COUNT(*) FROM images 
                    GROUP BY model_name
                """)
                model_distribution = dict(cursor.fetchall())
                
                # Get recent activity
                cursor.execute("""
                    SELECT COUNT(*) FROM images 
                    WHERE created_at >= datetime('now', '-7 days')
                """)
                recent_images = cursor.fetchone()[0]
                
                return {
                    'total_images': total_images,
                    'total_processing_time': total_processing_time,
                    'model_distribution': model_distribution,
                    'recent_images': recent_images,
                    'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024)
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def _row_to_image_record(self, row: Tuple) -> ImageRecord:
        """Convert database row to ImageRecord object"""
        metadata = {}
        if row[13]:  # metadata column
            try:
                metadata = json.loads(row[13])
            except:
                metadata = {}
        
        return ImageRecord(
            id=row[0],
            filename=row[1],
            original_path=row[2],
            processed_path=row[3],
            model_name=row[4],
            scale_factor=row[5],
            processing_time=row[6],
            file_size_original=row[7],
            file_size_processed=row[8],
            image_width=row[9],
            image_height=row[10],
            created_at=row[11],
            updated_at=row[12],
            metadata=metadata
        )
    
    def close(self):
        """Close database connection"""
        pass  # SQLite connections are automatically closed
