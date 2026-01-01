import os
import cv2
import uuid
import json
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
from flask_socketio import SocketIO, emit
import threading
import time
from functools import wraps
import base64

from config import db_config
from model_loader import model_loader  # Fixed import

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize SocketIO with eventlet
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
current_bus_turn = None
active_passengers = {}
camera_lock = threading.Lock()


def log_event(event_type, description, metadata=None):
    """Log system event to MongoDB"""
    db_config.log_event(event_type, description, metadata)


def requires_bus_turn(f):
    """Decorator to ensure bus turn is active"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_bus_turn is None:
            return jsonify({'error': 'No active bus turn. Start a journey first.'}), 400
        return f(*args, **kwargs)

    return decorated_function


@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')


@app.route('/start_journey', methods=['POST'])
def start_journey():
    """Start a new bus turn"""
    global current_bus_turn

    bus_turn_id = f"bus_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    current_bus_turn = {
        'bus_turn_id': bus_turn_id,
        'start_time': datetime.now(),
        'active': True,
        'passenger_count': 0,
        'anomaly_count': 0,
        'end_time': None
    }

    log_event('journey_started', f'Bus turn {bus_turn_id} started')

    socketio.emit('bus_turn_started', {
        'bus_turn_id': bus_turn_id,
        'start_time': current_bus_turn['start_time'].isoformat()
    })

    return jsonify({
        'success': True,
        'bus_turn_id': bus_turn_id,
        'start_time': current_bus_turn['start_time'].isoformat()
    })


@app.route('/capture_entrance/<passenger_id>', methods=['GET', 'POST'])
@requires_bus_turn
def capture_entrance(passenger_id):
    """Capture 4 entrance images within 5 seconds"""
    if request.method == 'GET':
        return render_template('capture.html',
                               passenger_id=passenger_id,
                               capture_type='entrance',
                               image_count=4)

    # POST request - handle image capture
    try:
        # Create directory for passenger
        entrance_dir = os.path.join('static', 'Entrance', passenger_id)
        os.makedirs(entrance_dir, exist_ok=True)

        images_data = []
        image_paths = []

        if 'images[]' in request.files:
            files = request.files.getlist('images[]')

            for i, file in enumerate(files[:4]):  # Only process first 4 images
                if file and file.filename:
                    filename = f"entrance_{i + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                    filepath = os.path.join(entrance_dir, filename)
                    file.save(filepath)

                    # Read image for processing
                    img = cv2.imread(filepath)
                    if img is not None:
                        images_data.append(img)
                        image_paths.append(filepath)

        # If no files uploaded, try base64 data
        elif 'image_data[]' in request.form:
            try:
                image_data_list = json.loads(request.form['image_data[]'])

                for i, image_data in enumerate(image_data_list[:4]):
                    if image_data:
                        # Decode base64 image
                        if ',' in image_data:
                            header, encoded = image_data.split(",", 1)
                        else:
                            encoded = image_data

                        binary_data = base64.b64decode(encoded)

                        filename = f"entrance_{i + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                        filepath = os.path.join(entrance_dir, filename)

                        # Save image
                        with open(filepath, 'wb') as f:
                            f.write(binary_data)

                        # Read image for processing
                        img = cv2.imread(filepath)
                        if img is not None:
                            images_data.append(img)
                            image_paths.append(filepath)
            except Exception as e:
                log_event('image_decode_error', f'Error decoding base64 images: {str(e)}')

        if not images_data:
            return jsonify({'error': 'No images captured'}), 400

        # Store in database
        journey_id = f"journey_{passenger_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Create or update passenger record
        passenger_name = request.form.get('name', 'Unknown')
        passenger_data = {
            'passenger_id': passenger_id,
            'name': passenger_name,
            'total_journeys': 0,
            'last_seen': datetime.now(),
            'created_at': datetime.now()
        }

        db_config.db.passengers.update_one(
            {'passenger_id': passenger_id},
            {'$setOnInsert': passenger_data, '$inc': {'total_journeys': 1}},
            upsert=True
        )

        # Create journey record
        journey_data = {
            'journey_id': journey_id,
            'bus_turn_id': current_bus_turn['bus_turn_id'],
            'passenger_id': passenger_id,
            'entrance_time': datetime.now(),
            'date': datetime.now().date(),
            'exit_time': None,
            'travel_time_seconds': None,
            'status': 'active'
        }

        db_config.db.journeys.insert_one(journey_data)

        # Store image metadata
        for i, img_path in enumerate(image_paths):
            image_data = {
                'image_id': str(uuid.uuid4()),
                'passenger_id': passenger_id,
                'image_type': 'entrance',
                'image_path': img_path.replace('static/', ''),
                'timestamp': datetime.now(),
                'journey_id': journey_id,
                'sequence': i + 1
            }
            db_config.db.images.insert_one(image_data)

        # Add to active passengers
        active_passengers[passenger_id] = {
            'journey_id': journey_id,
            'entrance_time': datetime.now(),
            'entrance_images': image_paths,
            'name': passenger_name
        }

        current_bus_turn['passenger_count'] += 1

        log_event('passenger_entered', f'Passenger {passenger_id} entered', {
            'passenger_id': passenger_id,
            'journey_id': journey_id,
            'name': passenger_name
        })

        socketio.emit('passenger_entered', {
            'passenger_id': passenger_id,
            'name': passenger_name,
            'journey_id': journey_id,
            'timestamp': datetime.now().isoformat()
        })

        return jsonify({
            'success': True,
            'message': f'{len(images_data)} entrance images captured for passenger {passenger_id}',
            'journey_id': journey_id,
            'image_count': len(images_data)
        })

    except Exception as e:
        log_event('capture_error', f'Error capturing entrance images: {str(e)}', {
            'passenger_id': passenger_id,
            'error': str(e)
        })
        return jsonify({'error': str(e)}), 500


@app.route('/capture_exit/<passenger_id>', methods=['GET', 'POST'])
@requires_bus_turn
def capture_exit(passenger_id):
    """Capture exit image and run anomaly detection"""
    if passenger_id not in active_passengers:
        return jsonify({'error': 'Passenger not found in active journeys'}), 404

    if request.method == 'GET':
        return render_template('capture.html',
                               passenger_id=passenger_id,
                               capture_type='exit',
                               image_count=1)

    try:
        # Create directory for exit image
        exit_dir = os.path.join('static', 'Exit', passenger_id)
        os.makedirs(exit_dir, exist_ok=True)

        exit_image = None
        exit_path = None

        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                filename = f"exit_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                exit_path = os.path.join(exit_dir, filename)
                file.save(exit_path)

                # Read image for processing
                exit_image = cv2.imread(exit_path)

        # Try base64 data
        elif 'image_data' in request.form:
            image_data = request.form['image_data']
            if image_data:
                try:
                    # Decode base64 image
                    if ',' in image_data:
                        header, encoded = image_data.split(",", 1)
                    else:
                        encoded = image_data

                    binary_data = base64.b64decode(encoded)

                    filename = f"exit_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                    exit_path = os.path.join(exit_dir, filename)

                    # Save image
                    with open(exit_path, 'wb') as f:
                        f.write(binary_data)

                    # Read image for processing
                    exit_image = cv2.imread(exit_path)
                except Exception as e:
                    log_event('exit_image_decode_error', f'Error decoding exit image: {str(e)}')

        if exit_image is None:
            return jsonify({'error': 'Failed to capture exit image'}), 400

        # Load entrance images
        entrance_images = []
        entrance_paths = active_passengers[passenger_id]['entrance_images']

        for img_path in entrance_paths:
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    entrance_images.append(img)

        if not entrance_images:
            return jsonify({'error': 'No entrance images found'}), 400

        # Run anomaly detection
        detection_result = model_loader.detect_anomaly(entrance_images, exit_image)

        # Get GAN anomaly score
        gan_score = model_loader.generate_anomaly_score_gan(exit_image)

        # Combine scores (weighted average)
        final_confidence = (detection_result['confidence'] * 0.7 + gan_score * 0.3)

        # Store exit image metadata
        image_data = {
            'image_id': str(uuid.uuid4()),
            'passenger_id': passenger_id,
            'image_type': 'exit',
            'image_path': exit_path.replace('static/', ''),
            'timestamp': datetime.now(),
            'journey_id': active_passengers[passenger_id]['journey_id'],
            'sequence': 0
        }
        db_config.db.images.insert_one(image_data)

        # Update journey record
        exit_time = datetime.now()
        entrance_time = active_passengers[passenger_id]['entrance_time']
        travel_time = int((exit_time - entrance_time).total_seconds())

        db_config.db.journeys.update_one(
            {'journey_id': active_passengers[passenger_id]['journey_id']},
            {
                '$set': {
                    'exit_time': exit_time,
                    'travel_time_seconds': travel_time,
                    'status': 'completed'
                }
            }
        )

        # Update passenger last seen
        db_config.db.passengers.update_one(
            {'passenger_id': passenger_id},
            {'$set': {'last_seen': exit_time}}
        )

        # Create alert record
        alert_type = 'anomaly' if detection_result['is_anomaly'] else 'normal'

        alert_data = {
            'alert_id': str(uuid.uuid4()),
            'passenger_id': passenger_id,
            'journey_id': active_passengers[passenger_id]['journey_id'],
            'alert_type': alert_type,
            'confidence': final_confidence,
            'timestamp': datetime.now(),
            'image_paths': [p.replace('static/', '') for p in entrance_paths + [exit_path]],
            'similarity_scores': detection_result['similarity_scores'],
            'alert_level': detection_result['alert_level']
        }

        alert_result = db_config.db.alerts.insert_one(alert_data)

        if detection_result['is_anomaly']:
            current_bus_turn['anomaly_count'] += 1

        # Remove from active passengers
        passenger_data = active_passengers.pop(passenger_id)

        # Log event
        log_event('passenger_exited', f'Passenger {passenger_id} exited', {
            'passenger_id': passenger_id,
            'anomaly': detection_result['is_anomaly'],
            'confidence': final_confidence,
            'travel_time': travel_time
        })

        # Emit real-time update
        socketio.emit('passenger_exit', {
            'passenger_id': passenger_id,
            'name': passenger_data['name'],
            'anomaly': detection_result['is_anomaly'],
            'confidence': final_confidence,
            'alert_level': detection_result['alert_level'],
            'travel_time': travel_time,
            'timestamp': datetime.now().isoformat(),
            'alert_id': str(alert_result.inserted_id)
        })

        return jsonify({
            'success': True,
            'anomaly': detection_result['is_anomaly'],
            'confidence': final_confidence,
            'similarity_scores': detection_result['similarity_scores'],
            'alert_level': detection_result['alert_level'],
            'travel_time': travel_time,
            'redirect': url_for('show_result', passenger_id=passenger_id)
        })

    except Exception as e:
        log_event('exit_error', f'Error processing exit: {str(e)}', {
            'passenger_id': passenger_id,
            'error': str(e)
        })
        return jsonify({'error': str(e)}), 500


@app.route('/result/<passenger_id>')
def show_result(passenger_id):
    """Display anomaly detection result"""
    # Get latest alert for this passenger
    alert = db_config.db.alerts.find_one(
        {'passenger_id': passenger_id},
        sort=[('timestamp', -1)]
    )

    if not alert:
        return render_template('result.html',
                               passenger={'passenger_id': passenger_id},
                               alert=None,
                               journey=None,
                               images=[])

    # Get journey details
    journey = db_config.db.journeys.find_one(
        {'journey_id': alert['journey_id']}
    )

    # Get passenger details
    passenger = db_config.db.passengers.find_one(
        {'passenger_id': passenger_id}
    )

    # Get images for this journey
    images = list(db_config.db.images.find(
        {'journey_id': alert['journey_id']},
        sort=[('sequence', 1)]
    ))

    return render_template('result.html',
                           alert=alert,
                           journey=journey,
                           passenger=passenger,
                           images=images)


@app.route('/dashboard')
def dashboard():
    """Real-time dashboard"""
    # Get recent alerts
    recent_alerts = list(db_config.db.alerts.find(
        sort=[('timestamp', -1)],
        limit=10
    ))

    # Get statistics
    total_passengers = db_config.db.passengers.count_documents({})
    total_journeys = db_config.db.journeys.count_documents({})
    total_alerts = db_config.db.alerts.count_documents({'alert_type': 'anomaly'})

    # Get today's stats
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_journeys = db_config.db.journeys.count_documents({
        'date': {'$gte': today_start}
    })

    # Get active journeys
    active_journeys = list(db_config.db.journeys.find(
        {'status': 'active'},
        sort=[('entrance_time', -1)]
    ))

    return render_template('dashboard.html',
                           current_bus_turn=current_bus_turn,
                           active_passengers=len(active_passengers),
                           recent_alerts=recent_alerts,
                           total_passengers=total_passengers,
                           total_journeys=total_journeys,
                           total_alerts=total_alerts,
                           today_journeys=today_journeys,
                           active_journeys=active_journeys)


@app.route('/history')
def history():
    """Passenger travel history"""
    passenger_id = request.args.get('passenger_id')
    date_filter = request.args.get('date')
    alert_type = request.args.get('alert_type')

    query = {}
    if passenger_id:
        query['passenger_id'] = passenger_id
    if date_filter:
        try:
            filter_date = datetime.strptime(date_filter, '%Y-%m-%d')
            query['date'] = filter_date.date()
        except:
            pass

    # Get journeys with their alerts
    journeys = list(db_config.db.journeys.find(
        query,
        sort=[('entrance_time', -1)],
        limit=50
    ))

    # Get alerts for these journeys
    for journey in journeys:
        alert = db_config.db.alerts.find_one(
            {'journey_id': journey['journey_id']}
        )
        journey['alert'] = alert

    # Filter by alert type
    if alert_type:
        journeys = [j for j in journeys if j.get('alert') and j['alert'].get('alert_type') == alert_type]

    # Get unique passenger IDs for dropdown
    passenger_ids = db_config.db.passengers.distinct('passenger_id')

    return render_template('history.html',
                           journeys=journeys,
                           passenger_ids=passenger_ids,
                           filters={'passenger_id': passenger_id,
                                    'date': date_filter,
                                    'alert_type': alert_type})


@app.route('/end_journey', methods=['POST'])
def end_journey():
    """End current bus turn and generate report"""
    global current_bus_turn

    if current_bus_turn is None:
        return jsonify({'error': 'No active bus turn'}), 400

    try:
        end_time = datetime.now()
        current_bus_turn['end_time'] = end_time
        current_bus_turn['active'] = False

        # Generate report
        report = {
            'bus_turn_id': current_bus_turn['bus_turn_id'],
            'start_time': current_bus_turn['start_time'],
            'end_time': end_time,
            'duration_seconds': int((end_time - current_bus_turn['start_time']).total_seconds()),
            'total_passengers': current_bus_turn['passenger_count'],
            'anomalies_detected': current_bus_turn['anomaly_count'],
            'active_passengers_remaining': len(active_passengers)
        }

        # Log event
        log_event('journey_ended', f'Bus turn {current_bus_turn["bus_turn_id"]} ended', report)

        # Clear active passengers
        active_passengers.clear()

        # Emit socket event
        socketio.emit('journey_ended', report)

        # Reset bus turn
        ended_turn = current_bus_turn.copy()
        current_bus_turn = None

        return jsonify({
            'success': True,
            'report': report,
            'message': 'Bus turn ended successfully'
        })

    except Exception as e:
        log_event('end_journey_error', f'Error ending journey: {str(e)}')
        return jsonify({'error': str(e)}), 500


# API Endpoints
@app.route('/api/passenger_history/<passenger_id>')
def get_passenger_history(passenger_id):
    """Get passenger history API endpoint"""
    journeys = list(db_config.db.journeys.find(
        {'passenger_id': passenger_id},
        sort=[('entrance_time', -1)]
    ))

    # Convert ObjectId to string for JSON serialization
    for journey in journeys:
        journey['_id'] = str(journey['_id'])
        if journey.get('entrance_time'):
            journey['entrance_time'] = journey['entrance_time'].isoformat()
        if journey.get('exit_time'):
            journey['exit_time'] = journey['exit_time'].isoformat()
        if journey.get('date'):
            journey['date'] = journey['date'].isoformat()

    return jsonify({'journeys': journeys})


@app.route('/api/recent_alerts')
def get_recent_alerts():
    """Get recent alerts API endpoint"""
    alerts = list(db_config.db.alerts.find(
        sort=[('timestamp', -1)],
        limit=20
    ))

    for alert in alerts:
        alert['_id'] = str(alert['_id'])
        alert['timestamp'] = alert['timestamp'].isoformat()

    return jsonify({'alerts': alerts})


@app.route('/api/dashboard_stats')
def get_dashboard_stats():
    """Get dashboard statistics"""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    stats = {
        'total_passengers': db_config.db.passengers.count_documents({}),
        'today_journeys': db_config.db.journeys.count_documents({'date': {'$gte': today_start}}),
        'anomalies': db_config.db.alerts.count_documents({'alert_type': 'anomaly'}),
        'active_passengers': len(active_passengers)
    }

    return jsonify(stats)


@app.route('/api/active_passengers')
def get_active_passengers():
    """Get active passengers"""
    return jsonify({
        'count': len(active_passengers),
        'passengers': list(active_passengers.keys())
    })


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print('Client connected')
    emit('connection_established', {'data': 'Connected to anomaly detection system'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print('Client disconnected')


# Camera feed simulation
def camera_feed():
    """Generate camera feed for dashboard"""
    # Create a simple test pattern
    while True:
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw some shapes to simulate camera feed
        cv2.rectangle(img, (50, 50), (590, 430), (0, 255, 0), 2)
        cv2.putText(img, "Camera Feed", (220, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, current_time, (220, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if current_bus_turn:
            cv2.putText(img, f"Bus: {current_bus_turn['bus_turn_id']}", (220, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)


@app.route('/video_feed')
def video_feed():
    """Video feed endpoint"""
    return Response(camera_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Additional routes
@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


@app.route('/capture')
def capture():
    """Capture page"""
    passenger_id = request.args.get('passenger_id', 'new')
    capture_type = request.args.get('type', 'entrance')
    return render_template('capture.html',
                           passenger_id=passenger_id,
                           capture_type=capture_type,
                           image_count=4 if capture_type == 'entrance' else 1)


@app.route('/api/check_passenger/<passenger_id>')
def check_passenger(passenger_id):
    """Check if passenger exists"""
    passenger = db_config.db.passengers.find_one({'passenger_id': passenger_id})
    if passenger:
        return jsonify({
            'exists': True,
            'name': passenger.get('name', 'Unknown'),
            'total_journeys': passenger.get('total_journeys', 0),
            'last_seen': passenger.get('last_seen').isoformat() if passenger.get('last_seen') else None
        })
    return jsonify({'exists': False})


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Connect to MongoDB
    if db_config.connect():
        print("Database connected successfully")
        log_event('system_start', 'Flask application started')

        # Ensure directories exist
        os.makedirs('static/Entrance', exist_ok=True)
        os.makedirs('static/Exit', exist_ok=True)
        os.makedirs('static/css', exist_ok=True)
        os.makedirs('static/js', exist_ok=True)
        os.makedirs('templates', exist_ok=True)

        # Run the application
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
    else:
        print("Failed to connect to database")