import cv2
import face_recognition
import numpy as np
import os
import pickle
import time
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ---------------- #
KNOWN_FACES_FILE = "known_faces.pkl"
MODEL_FILE = "face_model.pkl"
DIST_THRESHOLD = 0.5
PROB_THRESHOLD = 0.6
USE_CNN = False  

# -------------- LOGGING ----------------- #
logging.basicConfig(
    filename="face_system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------- UTILITIES ---------------- #

def load_known_faces():
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}


def save_known_faces(data):
    with open(KNOWN_FACES_FILE, "wb") as f:
        pickle.dump(data, f)


def train_model(known_faces):
    if not known_faces["encodings"]:
        return None, None

    X = np.array(known_faces["encodings"])
    y = np.array(known_faces["names"])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(50, X.shape[0]-1))),
        ("svc", SVC(kernel="rbf", probability=True))
    ])

    pipeline.fit(X, y_encoded)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": pipeline, "le": le}, f)

    logging.info("Model trained successfully")
    return pipeline, le


def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            data = pickle.load(f)
            return data["model"], data["le"]
    return None, None


# ---------------- LIVENESS ---------------- #

def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)


def check_liveness(frame, face_location):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks = face_recognition.face_landmarks(rgb, [face_location])

    if not landmarks:
        return False

    lm = landmarks[0]
    if "left_eye" in lm and "right_eye" in lm:
        left_ear = eye_aspect_ratio(lm["left_eye"])
        right_ear = eye_aspect_ratio(lm["right_eye"])
        ear = (left_ear + right_ear) / 2.0

        return ear > 0.2  # threshold
    return False


# --------------- EVALUATION -------------- #

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve (AUC={roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


# -------------- MAIN SYSTEM -------------- #

def main():
    known_faces = load_known_faces()
    model, le = load_model()

    if model is None and known_faces["encodings"]:
        model, le = train_model(known_faces)

    video = cv2.VideoCapture(0)
    prev_time = 0

    print("Press 'e' to enroll, 'q' to quit")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # FPS Counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="cnn" if USE_CNN else "hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for (top, right, bottom, left), encoding in zip(boxes, encodings):

            name = "Unknown"
            color = (0, 0, 255)

            # Liveness Check
            if not check_liveness(frame, (top, right, bottom, left)):
                name = "Spoof!"
                color = (255, 0, 0)
                logging.warning("Spoof attempt detected")
            else:
                if model:
                    probs = model.predict_proba([encoding])[0]
                    best_idx = np.argmax(probs)
                    if probs[best_idx] > PROB_THRESHOLD:
                        name = le.inverse_transform([best_idx])[0]
                        color = (0, 255, 0)
                        logging.info(f"Recognized: {name}")

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1)
        if key == ord('e'):
            name = input("Enter Name: ")
            if encodings:
                known_faces["encodings"].append(encodings[0])
                known_faces["names"].append(name)
                save_known_faces(known_faces)
                model, le = train_model(known_faces)

        elif key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
