import os
import json
import base64
import boto3
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import tensorflow as tf
from PIL import Image
from io import BytesIO
from facenet_pytorch import MTCNN

# ---------------------- Flask Setup ---------------------- #
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Get port from environment variable (for App Runner)
PORT = int(os.getenv("PORT", 5000))

# ---------------------- AWS S3 Setup ---------------------- #
S3_BUCKET = os.getenv("S3_BUCKET", "mobilefacenet")
s3 = boto3.client("s3")

# ---------------------- TFLite Model Setup ---------------------- #
print("Loading TFLite face recognition model...")

# Load TFLite model
try:
    tflite_model_path = os.path.join(os.path.dirname(__file__), "face_recognition_model.tflite")
    
    if os.path.exists(tflite_model_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        print("✅ TFLite model loaded successfully!")
    else:
        print("⚠️  TFLite model file not found. Please provide 'face_recognition_model.tflite'")
        interpreter = None
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")
    interpreter = None

# Load face detector
print("Loading face detector...")
face_detector = MTCNN(image_size=160, margin=20, device='cpu')
print("✅ Face detector loaded successfully!")

# ---------------------- Helper Functions ---------------------- #
def decode_image(base64_str):
    try:
        print(f"🔍 Decoding base64 image, length: {len(base64_str)}")
        
        # Remove data URL prefix if present
        if base64_str.startswith('data:image/'):
            base64_str = base64_str.split(',')[1]
            print(f"✅ Removed data URL prefix, base64 length: {len(base64_str)}")
        
        img_bytes = base64.b64decode(base64_str)
        print(f"✅ Base64 decoded, bytes: {len(img_bytes)}")
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        print(f"✅ PIL image created, size: {img.size}, mode: {img.mode}")
        return img
    except Exception as e:
        print(f"❌ Image decoding failed: {e}")
        return None

def extract_face_embedding(base64_str):
    """Extract face embedding using TFLite model"""
    try:
        if interpreter is None:
            print("❌ TFLite interpreter is None")
            return None
            
        img = decode_image(base64_str)
        if img is None:
            print("❌ Failed to decode image from base64")
            return None

        print(f"✅ Image decoded successfully, size: {img.size}")
        
        # Detect and align face using MTCNN
        print("🔍 Detecting face with MTCNN...")
        face = face_detector(img)
        if face is None:
            print("❌ No face detected by MTCNN")
            return None
            
        print("✅ Face detected successfully")
        print(f"🔍 MTCNN face tensor shape: {face.shape}")

        # Get input details from TFLite model
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        
        print(f"🔍 Model input shape: {input_shape}")
        
        # Convert MTCNN tensor to numpy array and fix dimensions
        if hasattr(face, 'shape'):
            # MTCNN returns (channels, height, width) - convert to (height, width, channels)
            if len(face.shape) == 3 and face.shape[0] == 3:
                # Transpose from (3, H, W) to (H, W, 3)
                face_array = face.permute(1, 2, 0).numpy()
                print(f"🔍 Transposed face array shape: {face_array.shape}")
            else:
                face_array = face.numpy()
        else:
            # If it's already a PIL image or numpy array
            face_array = np.array(face)
        
        # Resize if needed to match model input dimensions
        target_height = input_shape[1]
        target_width = input_shape[2]
        
        if face_array.shape[:2] != (target_height, target_width):
            # Use PIL resize for better control
            face_pil = Image.fromarray(face_array.astype(np.uint8))
            face_resized = face_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
            face_array = np.array(face_resized)
            print(f"🔍 Resized face array shape: {face_array.shape}")
        
        # Normalize to [0, 1] range
        face_array = face_array.astype(np.float32) / 255.0
        
        # Ensure correct shape: (batch, height, width, channels)
        if len(face_array.shape) == 3:
            face_input = np.expand_dims(face_array, axis=0)
        else:
            face_input = face_array
            
        print(f"🔍 Final input shape: {face_input.shape}")
        
        # Validate input tensor shape matches model expectation
        expected_shape = tuple(input_shape)
        if face_input.shape != expected_shape:
            print(f"❌ Shape mismatch! Expected: {expected_shape}, Got: {face_input.shape}")
            # Try to reshape if possible
            try:
                face_input = face_input.reshape(expected_shape)
                print(f"✅ Reshaped to expected shape: {face_input.shape}")
            except Exception as reshape_error:
                print(f"❌ Failed to reshape: {reshape_error}")
                return None
        
        # Set input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], face_input)
        interpreter.invoke()
        
        # Get output tensor
        output_details = interpreter.get_output_details()
        embedding = interpreter.get_tensor(output_details[0]['index'])
        
        return embedding.flatten()
        
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def save_to_s3(key, data, content_type="application/json"):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)

def read_from_s3(key):
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        data = obj["Body"].read()
        return data
    except Exception as e:
        print(f"ERROR: Failed to read from S3: {str(e)}")
        raise e

# ---------------------- Routes ---------------------- #
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"success": True, "message": "Face verification API running!"}), 200

@app.route("/register", methods=["POST"])
def register_user():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        username = data.get("username")
        email = data.get("email")
        images = data.get("images", [])

        if not user_id or not username or not email or not images:
            return jsonify({"success": False, "message": "User ID, username, email and images required"}), 400

        if len(images) < 1:
            return jsonify({"success": False, "message": "At least 1 image required for registration"}), 400

        # Process the image and extract embedding
        embedding = extract_face_embedding(images[0])
        if embedding is None:
            return jsonify({"success": False, "message": "Failed to extract face embedding from the image"}), 400

        embedding = normalize_embedding(embedding)

        # Save the profile image
        img_bytes = base64.b64decode(images[0])
        image_key = f"{user_id}/profile_image.jpg"
        save_to_s3(image_key, img_bytes, "image/jpeg")

        # Save metadata JSON with the embedding
        metadata = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "created_at": datetime.utcnow().isoformat(),
            "total_images": 1,
            "image_keys": ["profile_image.jpg"],
            "embedding": embedding.tolist(),
            "individual_embeddings": [embedding.tolist()]
        }
        
        save_to_s3(f"{user_id}/user_details.json", json.dumps(metadata), "application/json")

        return jsonify({
            "success": True, 
            "message": "User registered successfully with 1 image",
            "images_processed": 1,
            "total_images": len(images)
        }), 201

    except Exception as e:
        print(f"ERROR: Registration failed: {str(e)}")
        return jsonify({"success": False, "message": f"Registration failed: {str(e)}"}), 500

@app.route("/verify", methods=["POST"])
def verify_user():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        live_b64 = data.get("live_image")

        if not user_id or not live_b64:
            return jsonify({"success": False, "message": "Missing user_id or live image"}), 400

        # Read stored embedding from JSON
        stored_json = read_from_s3(f"{user_id}/user_details.json")
        metadata = json.loads(stored_json)
        
        # Get all individual embeddings for robust verification
        individual_embeddings = metadata.get("individual_embeddings", [])
        if not individual_embeddings:
            stored_embedding = np.array(metadata["embedding"], dtype=np.float32)
            individual_embeddings = [stored_embedding]
        
        # Extract face embedding from live image
        live_embedding = extract_face_embedding(live_b64)
        if live_embedding is None:
            return jsonify({"success": False, "message": "No face detected in live image"}), 400

        live_embedding = normalize_embedding(live_embedding)
        
        # Compare against all embeddings for this user
        best_similarity = -1.0
        best_embedding_index = -1
        
        for i, stored_emb in enumerate(individual_embeddings):
            stored_embedding = np.array(stored_emb, dtype=np.float32)
            stored_embedding = normalize_embedding(stored_embedding)
            
            # Calculate cosine similarity
            dot_product = np.dot(stored_embedding, live_embedding)
            norm_stored = np.linalg.norm(stored_embedding)
            norm_live = np.linalg.norm(live_embedding)
            
            if norm_stored > 0 and norm_live > 0:
                similarity = dot_product / (norm_stored * norm_live)
            else:
                similarity = 0.0
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_embedding_index = i

        # Check if the best match meets the threshold
        if best_similarity >= 0.6:
            return jsonify({
                "success": True,
                "message": "Face verified successfully",
                "similarity": float(best_similarity),
                "matched_embedding": best_embedding_index + 1,
                "total_embeddings": len(individual_embeddings),
                "user": {
                    "user_id": metadata["user_id"]
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Face not recognized",
                "similarity": float(best_similarity),
                "threshold": 0.6,
                "total_embeddings_checked": len(individual_embeddings)
            }), 401

    except Exception as e:
        print(f"ERROR: Verification failed: {str(e)}")
        return jsonify({"success": False, "message": f"Verification failed: {str(e)}"}), 500

@app.route("/identify", methods=["POST"])
def identify_user_by_face():
    """Identify user by face image without requiring user_id"""
    try:
        data = request.get_json()
        face_b64 = data.get("face_image")

        if not face_b64:
            return jsonify({"success": False, "message": "Face image is required"}), 400

        # Extract face embedding from the provided image
        face_embedding = extract_face_embedding(face_b64)
        if face_embedding is None:
            return jsonify({"success": False, "message": "No face detected in the image"}), 400

        face_embedding = normalize_embedding(face_embedding)

        # List all users in S3 and compare against each one
        best_match = None
        best_similarity = -1.0
        best_user_id = None

        try:
            response = s3.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=1000)
            objects = response.get('Contents', [])
            
            # Group by user_id (folder structure)
            users = {}
            for obj in objects:
                key = obj['Key']
                if '/' in key:
                    user_id = key.split('/')[0]
                    if user_id not in users:
                        users[user_id] = []
                    users[user_id].append(key)

            # Check each user's embeddings
            for user_id in users.keys():
                try:
                    stored_json = read_from_s3(f"{user_id}/user_details.json")
                    metadata = json.loads(stored_json)
                    
                    individual_embeddings = metadata.get("individual_embeddings", [])
                    if not individual_embeddings:
                        stored_embedding = np.array(metadata["embedding"], dtype=np.float32)
                        individual_embeddings = [stored_embedding]
                    
                    # Compare against all embeddings for this user
                    for i, stored_emb in enumerate(individual_embeddings):
                        stored_embedding = np.array(stored_emb, dtype=np.float32)
                        stored_embedding = normalize_embedding(stored_embedding)
                        
                        # Calculate cosine similarity
                        dot_product = np.dot(stored_embedding, face_embedding)
                        norm_stored = np.linalg.norm(stored_embedding)
                        norm_live = np.linalg.norm(face_embedding)
                        
                        if norm_stored > 0 and norm_live > 0:
                            similarity = dot_product / (norm_stored * norm_live)
                        else:
                            similarity = 0.0
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = metadata
                            best_user_id = user_id
                            
                except Exception as e:
                    continue

            # Check if the best match meets the threshold
            if best_similarity >= 0.6 and best_match is not None:
                return jsonify({
                    "success": True,
                    "message": "User identified successfully",
                    "similarity": float(best_similarity),
                    "user": {
                        "user_id": best_user_id,
                        "username": best_match.get("username", "User")
                    }
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "message": "No matching user found",
                    "similarity": float(best_similarity) if best_similarity > -1 else 0.0,
                    "threshold": 0.6
                }), 401

        except Exception as e:
            print(f"ERROR: Failed to list S3 objects: {str(e)}")
            return jsonify({"success": False, "message": f"Database access failed: {str(e)}"}), 500

    except Exception as e:
        print(f"ERROR: Face identification failed: {str(e)}")
        return jsonify({"success": False, "message": f"Face identification failed: {str(e)}"}), 500

@app.route("/valid", methods=["GET"])
def validate_user():
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"success": False, "message": "Missing user_id"}), 400

        try:
            read_from_s3(f"{user_id}/user_details.json")
            return jsonify({"success": True, "message": "User is valid"}), 200
        except Exception as e:
            return jsonify({"success": False, "message": f"User not found: {str(e)}"}), 404

    except Exception as e:
        return jsonify({"success": False, "message": f"Validation failed: {str(e)}"}), 500

@app.route("/logout", methods=["POST", "OPTIONS"])
def logout():
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200
    return jsonify({"success": True, "message": "User logged out successfully"}), 200

# ---------------------- Run ---------------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
