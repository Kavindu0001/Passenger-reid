from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBConfig:
    def __init__(self):
        # MongoDB connection
        self.mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        self.database_name = 'Passenger_Anomaly'
        self.client = None
        self.db = None

    def connect(self):
        """Establish MongoDB connection with auto-configuration"""
        try:
            self.client = MongoClient(self.mongo_uri,
                                      serverSelectionTimeoutMS=5000,
                                      connectTimeoutMS=10000,
                                      socketTimeoutMS=10000)

            # Test connection
            self.client.server_info()
            logger.info(f"Connected to MongoDB: {self.mongo_uri}")

            # Get or create database
            self.db = self.client[self.database_name]

            # Auto-configure collections
            self._auto_configure_database()

            # Create indexes
            self._create_indexes()

            return True

        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            logger.info("Attempting to start MongoDB service...")
            return self._start_mongodb_service()
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            return False

    def _start_mongodb_service(self):
        """Attempt to start MongoDB service automatically"""
        try:
            import platform
            import subprocess

            system = platform.system()

            if system == "Windows":
                # Try to start MongoDB service on Windows
                result = subprocess.run(['sc', 'start', 'MongoDB'],
                                        capture_output=True,
                                        text=True,
                                        timeout=30)
                if result.returncode == 0:
                    logger.info("MongoDB service started successfully")
                    time.sleep(3)  # Wait for service to start
                    return self.connect()
                else:
                    logger.error(f"Failed to start MongoDB service: {result.stderr}")

            elif system == "Linux" or system == "Darwin":  # Linux or macOS
                # Try to start MongoDB service
                result = subprocess.run(['sudo', 'systemctl', 'start', 'mongod'],
                                        capture_output=True,
                                        text=True,
                                        timeout=30)
                if result.returncode == 0:
                    logger.info("MongoDB service started successfully")
                    time.sleep(3)
                    return self.connect()
                else:
                    # Try mongodb command
                    result = subprocess.run(['mongod', '--dbpath', './data', '--fork', '--logpath', 'mongodb.log'],
                                            capture_output=True,
                                            text=True,
                                            timeout=30)
                    if result.returncode == 0:
                        logger.info("MongoDB started in background")
                        time.sleep(3)
                        return self.connect()

            logger.error("Could not start MongoDB automatically. Please start it manually.")
            return False

        except Exception as e:
            logger.error(f"Error starting MongoDB: {e}")
            return False

    def _auto_configure_database(self):
        """Auto-configure all collections with proper schemas"""
        logger.info("Auto-configuring database collections...")

        # Check if database exists
        database_names = self.client.list_database_names()
        if self.database_name not in database_names:
            logger.info(f"Creating new database: {self.database_name}")

        # Define collection schemas
        collections_config = {
            'passengers': self._get_passengers_schema(),
            'journeys': self._get_journeys_schema(),
            'images': self._get_images_schema(),
            'alerts': self._get_alerts_schema(),
            'system_logs': self._get_system_logs_schema()
        }

        for collection_name, schema in collections_config.items():
            self._configure_collection(collection_name, schema)

        logger.info("Database configuration completed")

    def _configure_collection(self, collection_name, schema):
        """Configure a single collection"""
        try:
            # Check if collection exists
            if collection_name in self.db.list_collection_names():
                logger.info(f"Collection '{collection_name}' already exists")
                return

            # Create collection with schema validation
            self.db.create_collection(collection_name, validator={
                '$jsonSchema': schema
            })

            logger.info(f"Created collection '{collection_name}' with schema validation")

        except OperationFailure as e:
            if e.code == 48:  # Collection already exists
                logger.info(f"Collection '{collection_name}' already exists")
            elif e.code == 73:  # Invalid schema
                logger.warning(f"Could not apply schema to '{collection_name}', creating without validation")
                if collection_name not in self.db.list_collection_names():
                    self.db.create_collection(collection_name)
            else:
                logger.error(f"Error creating collection '{collection_name}': {e}")
                # Create collection without validation as fallback
                if collection_name not in self.db.list_collection_names():
                    self.db.create_collection(collection_name)
        except Exception as e:
            logger.error(f"Unexpected error creating collection '{collection_name}': {e}")

    def _get_passengers_schema(self):
        """Get passengers collection schema"""
        return {
            'bsonType': 'object',
            'required': ['passenger_id', 'created_at'],
            'properties': {
                'passenger_id': {
                    'bsonType': 'string',
                    'description': 'must be a string and is required'
                },
                'name': {
                    'bsonType': 'string',
                    'description': 'must be a string'
                },
                'total_journeys': {
                    'bsonType': 'int',
                    'minimum': 0,
                    'description': 'must be a non-negative integer'
                },
                'last_seen': {
                    'bsonType': 'date',
                    'description': 'must be a date'
                },
                'created_at': {
                    'bsonType': 'date',
                    'description': 'must be a date and is required'
                },
                'contact_info': {
                    'bsonType': 'object',
                    'properties': {
                        'email': {'bsonType': 'string'},
                        'phone': {'bsonType': 'string'}
                    }
                },
                'metadata': {
                    'bsonType': 'object',
                    'description': 'additional metadata'
                }
            }
        }

    def _get_journeys_schema(self):
        """Get journeys collection schema"""
        return {
            'bsonType': 'object',
            'required': ['journey_id', 'bus_turn_id', 'passenger_id', 'entrance_time'],
            'properties': {
                'journey_id': {
                    'bsonType': 'string',
                    'description': 'must be a string and is required'
                },
                'bus_turn_id': {
                    'bsonType': 'string',
                    'description': 'must be a string and is required'
                },
                'passenger_id': {
                    'bsonType': 'string',
                    'description': 'must be a string and is required'
                },
                'entrance_time': {
                    'bsonType': 'date',
                    'description': 'must be a date and is required'
                },
                'exit_time': {
                    'bsonType': 'date',
                    'description': 'must be a date'
                },
                'travel_time_seconds': {
                    'bsonType': 'int',
                    'minimum': 0,
                    'description': 'must be a non-negative integer'
                },
                'date': {
                    'bsonType': 'date',
                    'description': 'must be a date'
                },
                'status': {
                    'bsonType': 'string',
                    'enum': ['active', 'completed', 'cancelled'],
                    'description': 'must be one of: active, completed, cancelled'
                },
                'anomaly_detected': {
                    'bsonType': 'bool',
                    'description': 'indicates if anomaly was detected'
                }
            }
        }

    def _get_images_schema(self):
        """Get images collection schema"""
        return {
            'bsonType': 'object',
            'required': ['image_id', 'passenger_id', 'image_type', 'timestamp'],
            'properties': {
                'image_id': {
                    'bsonType': 'string',
                    'description': 'must be a string and is required'
                },
                'passenger_id': {
                    'bsonType': 'string',
                    'description': 'must be a string and is required'
                },
                'image_type': {
                    'bsonType': 'string',
                    'enum': ['entrance', 'exit'],
                    'description': 'must be either "entrance" or "exit" and is required'
                },
                'image_path': {
                    'bsonType': 'string',
                    'description': 'must be a string'
                },
                'timestamp': {
                    'bsonType': 'date',
                    'description': 'must be a date and is required'
                },
                'journey_id': {
                    'bsonType': 'string',
                    'description': 'must be a string'
                },
                'sequence': {
                    'bsonType': 'int',
                    'minimum': 0,
                    'description': 'image sequence number'
                },
                'embeddings': {
                    'bsonType': 'array',
                    'items': {'bsonType': 'double'},
                    'description': 'feature embeddings array'
                }
            }
        }

    def _get_alerts_schema(self):
        """Get alerts collection schema"""
        return {
            'bsonType': 'object',
            'required': ['alert_id', 'passenger_id', 'alert_type', 'timestamp'],
            'properties': {
                'alert_id': {
                    'bsonType': 'string',
                    'description': 'must be a string and is required'
                },
                'passenger_id': {
                    'bsonType': 'string',
                    'description': 'must be a string and is required'
                },
                'journey_id': {
                    'bsonType': 'string',
                    'description': 'must be a string'
                },
                'alert_type': {
                    'bsonType': 'string',
                    'enum': ['anomaly', 'normal', 'warning', 'critical'],
                    'description': 'must be one of: anomaly, normal, warning, critical and is required'
                },
                'confidence': {
                    'bsonType': 'double',
                    'minimum': 0,
                    'maximum': 1,
                    'description': 'must be a double between 0 and 1'
                },
                'timestamp': {
                    'bsonType': 'date',
                    'description': 'must be a date and is required'
                },
                'image_paths': {
                    'bsonType': 'array',
                    'items': {'bsonType': 'string'},
                    'description': 'array of image paths'
                },
                'similarity_scores': {
                    'bsonType': 'array',
                    'items': {'bsonType': 'double'},
                    'description': 'array of similarity scores'
                },
                'alert_level': {
                    'bsonType': 'string',
                    'enum': ['low', 'medium', 'high'],
                    'description': 'must be one of: low, medium, high'
                },
                'resolved': {
                    'bsonType': 'bool',
                    'description': 'indicates if alert is resolved'
                },
                'resolved_at': {
                    'bsonType': 'date',
                    'description': 'when alert was resolved'
                }
            }
        }

    def _get_system_logs_schema(self):
        """Get system_logs collection schema"""
        return {
            'bsonType': 'object',
            'required': ['log_id', 'event_type', 'timestamp'],
            'properties': {
                'log_id': {
                    'bsonType': 'string',
                    'description': 'must be a string and is required'
                },
                'event_type': {
                    'bsonType': 'string',
                    'description': 'must be a string and is required'
                },
                'description': {
                    'bsonType': 'string',
                    'description': 'must be a string'
                },
                'timestamp': {
                    'bsonType': 'date',
                    'description': 'must be a date and is required'
                },
                'metadata': {
                    'bsonType': 'object',
                    'description': 'additional metadata'
                },
                'severity': {
                    'bsonType': 'string',
                    'enum': ['info', 'warning', 'error', 'critical'],
                    'description': 'log severity level'
                },
                'source': {
                    'bsonType': 'string',
                    'description': 'source of the log entry'
                }
            }
        }

    def _create_indexes(self):
        """Create indexes for faster queries"""
        logger.info("Creating database indexes...")

        try:
            # passengers collection indexes
            self.db.passengers.create_index([('passenger_id', 1)], unique=True, name='passenger_id_unique')
            self.db.passengers.create_index([('last_seen', -1)], name='last_seen_desc')
            self.db.passengers.create_index([('created_at', -1)], name='created_at_desc')

            # journeys collection indexes
            self.db.journeys.create_index([('journey_id', 1)], unique=True, name='journey_id_unique')
            self.db.journeys.create_index([('passenger_id', 1)], name='passenger_id_index')
            self.db.journeys.create_index([('bus_turn_id', 1)], name='bus_turn_id_index')
            self.db.journeys.create_index([('date', -1)], name='date_desc')
            self.db.journeys.create_index([('status', 1)], name='status_index')
            self.db.journeys.create_index([('entrance_time', -1)], name='entrance_time_desc')

            # images collection indexes
            self.db.images.create_index([('image_id', 1)], unique=True, name='image_id_unique')
            self.db.images.create_index([('passenger_id', 1)], name='image_passenger_id_index')
            self.db.images.create_index([('timestamp', -1)], name='image_timestamp_desc')
            self.db.images.create_index([('journey_id', 1)], name='image_journey_id_index')
            self.db.images.create_index([('image_type', 1)], name='image_type_index')

            # alerts collection indexes
            self.db.alerts.create_index([('alert_id', 1)], unique=True, name='alert_id_unique')
            self.db.alerts.create_index([('passenger_id', 1)], name='alert_passenger_id_index')
            self.db.alerts.create_index([('timestamp', -1)], name='alert_timestamp_desc')
            self.db.alerts.create_index([('alert_type', 1)], name='alert_type_index')
            self.db.alerts.create_index([('alert_level', 1)], name='alert_level_index')
            self.db.alerts.create_index([('resolved', 1)], name='alert_resolved_index')

            # system_logs collection indexes
            self.db.system_logs.create_index([('log_id', 1)], unique=True, name='log_id_unique')
            self.db.system_logs.create_index([('timestamp', -1)], name='log_timestamp_desc')
            self.db.system_logs.create_index([('event_type', 1)], name='event_type_index')
            self.db.system_logs.create_index([('severity', 1)], name='severity_index')

            logger.info("✓ Database indexes created successfully")

        except Exception as e:
            logger.error(f"Error creating indexes: {e}")

    def insert_sample_data(self):
        """Insert sample data for testing"""
        try:
            # Check if sample data already exists
            if self.db.passengers.count_documents({}) > 0:
                logger.info("Sample data already exists, skipping...")
                return

            logger.info("Inserting sample data...")

            from datetime import datetime, timedelta
            import uuid

            # Sample passengers
            sample_passengers = [
                {
                    'passenger_id': 'P001',
                    'name': 'John Doe',
                    'total_journeys': 5,
                    'last_seen': datetime.now() - timedelta(hours=2),
                    'created_at': datetime.now() - timedelta(days=30),
                    'contact_info': {'email': 'john@example.com', 'phone': '1234567890'}
                },
                {
                    'passenger_id': 'P002',
                    'name': 'Jane Smith',
                    'total_journeys': 3,
                    'last_seen': datetime.now() - timedelta(hours=5),
                    'created_at': datetime.now() - timedelta(days=25),
                    'contact_info': {'email': 'jane@example.com', 'phone': '0987654321'}
                },
                {
                    'passenger_id': 'P003',
                    'name': 'Robert Johnson',
                    'total_journeys': 8,
                    'last_seen': datetime.now() - timedelta(days=1),
                    'created_at': datetime.now() - timedelta(days=45),
                    'contact_info': {'email': 'robert@example.com', 'phone': '5551234567'}
                }
            ]

            self.db.passengers.insert_many(sample_passengers)

            # Sample journeys
            sample_journeys = []
            for i in range(10):
                journey_date = datetime.now() - timedelta(days=i)
                entrance_time = journey_date.replace(hour=8, minute=30, second=0)
                exit_time = entrance_time + timedelta(hours=1, minutes=30)

                sample_journeys.append({
                    'journey_id': f'J{str(1000 + i)}',
                    'bus_turn_id': f'BT{str(500 + i)}',
                    'passenger_id': f'P00{str((i % 3) + 1)}',
                    'entrance_time': entrance_time,
                    'exit_time': exit_time,
                    'travel_time_seconds': 5400,  # 1.5 hours
                    'date': journey_date.date(),
                    'status': 'completed',
                    'anomaly_detected': i % 4 == 0  # Every 4th journey has anomaly
                })

            self.db.journeys.insert_many(sample_journeys)

            # Sample alerts
            sample_alerts = []
            for i in range(5):
                alert_time = datetime.now() - timedelta(hours=i * 2)
                sample_alerts.append({
                    'alert_id': f'AL{str(100 + i)}',
                    'passenger_id': f'P00{str((i % 3) + 1)}',
                    'journey_id': f'J{str(1000 + i)}',
                    'alert_type': 'anomaly' if i % 2 == 0 else 'normal',
                    'confidence': 0.85 if i % 2 == 0 else 0.95,
                    'timestamp': alert_time,
                    'image_paths': [f'static/Entrance/P00{str((i % 3) + 1)}/image_{j}.jpg' for j in range(4)],
                    'similarity_scores': [0.75, 0.82, 0.79, 0.85] if i % 2 == 0 else [0.92, 0.94, 0.91, 0.93],
                    'alert_level': 'medium' if i % 2 == 0 else 'low',
                    'resolved': i > 2,
                    'resolved_at': alert_time + timedelta(minutes=30) if i > 2 else None
                })

            self.db.alerts.insert_many(sample_alerts)

            # Sample system logs
            sample_logs = []
            log_types = ['system_start', 'passenger_entered', 'passenger_exited', 'anomaly_detected',
                         'bus_turn_started']
            severities = ['info', 'info', 'info', 'warning', 'info']

            for i in range(20):
                log_time = datetime.now() - timedelta(minutes=i * 15)
                log_type = log_types[i % len(log_types)]

                sample_logs.append({
                    'log_id': f'LOG{str(1000 + i)}',
                    'event_type': log_type,
                    'description': f'Sample {log_type} event #{i}',
                    'timestamp': log_time,
                    'severity': severities[i % len(severities)],
                    'source': 'system',
                    'metadata': {
                        'iteration': i,
                        'auto_generated': True,
                        'test_data': True
                    }
                })

            self.db.system_logs.insert_many(sample_logs)

            logger.info("✓ Sample data inserted successfully")

        except Exception as e:
            logger.error(f"Error inserting sample data: {e}")

    def get_database_stats(self):
        """Get database statistics"""
        try:
            stats = {
                'database': self.database_name,
                'collections': {},
                'total_size': 0
            }

            # Get collection stats
            for collection_name in self.db.list_collection_names():
                stats_command = {'collStats': collection_name}
                collection_stats = self.db.command(stats_command)

                stats['collections'][collection_name] = {
                    'count': collection_stats.get('count', 0),
                    'size': collection_stats.get('size', 0),
                    'storageSize': collection_stats.get('storageSize', 0),
                    'indexes': len(collection_stats.get('indexSizes', {}))
                }

                stats['total_size'] += collection_stats.get('size', 0)

            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return None

    def backup_database(self, backup_path='backups/'):
        """Create a backup of the database"""
        try:
            import shutil
            from datetime import datetime

            os.makedirs(backup_path, exist_ok=True)

            backup_file = os.path.join(
                backup_path,
                f'{self.database_name}_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )

            backup_data = {
                'database': self.database_name,
                'backup_time': datetime.now().isoformat(),
                'collections': {}
            }

            for collection_name in self.db.list_collection_names():
                documents = list(self.db[collection_name].find({}, {'_id': 0}))
                backup_data['collections'][collection_name] = documents

            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)

            logger.info(f"✓ Database backup created: {backup_file}")
            return backup_file

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None

    def log_event(self, event_type, description, metadata=None):
        """Log system event"""
        try:
            log_entry = {
                'log_id': f'log_{datetime.now().strftime("%Y%m%d%H%M%S%f")}',
                'event_type': event_type,
                'description': description,
                'timestamp': datetime.now(),
                'metadata': metadata or {},
                'severity': 'info',
                'source': 'application'
            }
            self.db.system_logs.insert_one(log_entry)
        except Exception as e:
            logger.error(f"Error logging event: {e}")

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Global MongoDB instance
db_config = MongoDBConfig()