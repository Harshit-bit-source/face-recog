import cv2
import face_recognition
import pickle
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

KNOWN_FACES_FILE = "known_faces.pkl"
MODEL_FILE = "face_model.pkl"
PROB_THRESHOLD = 0.65
DIST_THRESHOLD = 0.50

HIGH_ACCURACY_MODE = True
USE_PCA = True
PCA_MAX_COMPONENTS = 60
USE_CNN_DETECT = True

SHOW_HEAD_SQUARE = True
SQUARE_SCALE = 0.25
SQUARE_MARGIN = 8

def load_known_faces():
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def save_known_faces(data):
    with open(KNOWN_FACES_FILE, "wb") as f:
        pickle.dump(data, f)

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            data = pickle.load(f)
            return data.get("model"), data.get("le"), data.get("metrics", None), data.get("scaler", None), data.get("pca", None)
    return None, None, None, None, None

def evaluate_model_cv(X, y_enc, estimator=None):
    if len(X) < 2:
        return None
    unique, counts = np.unique(y_enc, return_counts=True)
    min_class_count = counts.min()
    n_splits = min(5, int(min_class_count))
    if n_splits < 2:
        return None
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        est = estimator if estimator is not None else KNeighborsClassifier(n_neighbors=min(3, len(X)))
        y_pred = cross_val_predict(est, X, y_enc, cv=skf, n_jobs=1)
        metrics = {
            "accuracy": float(accuracy_score(y_enc, y_pred)),
            "precision_macro": float(precision_score(y_enc, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_enc, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_enc, y_pred, average="macro", zero_division=0)),
            "cv_folds": int(n_splits),
        }
        return metrics
    except Exception:
        return None

def train_classifier(known_faces):
    if not known_faces["encodings"]:
        return None, None, None, None, None
    X = np.array(known_faces["encodings"])
    y = np.array(known_faces["names"])
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = None
    Xp = Xs
    if USE_PCA:
        from sklearn.decomposition import PCA
        n_components = min(PCA_MAX_COMPONENTS, Xs.shape[1], max(1, Xs.shape[0] - 1))
        if n_components >= 10:
            pca = PCA(n_components=n_components, random_state=42)
            Xp = pca.fit_transform(Xs)
    model = None
    metrics = None
    if len(X) < 30:
        n_neighbors = min(5, len(X))
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(Xp, y_enc)
        metrics = evaluate_model_cv(Xp, y_enc, estimator=model)
    else:
        from sklearn.svm import SVC
        if HIGH_ACCURACY_MODE:
            param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
            base = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
            gs = GridSearchCV(base, param_grid, cv=StratifiedKFold(n_splits=min(4, int(np.min(np.unique(y_enc, return_counts=True)[1])))), n_jobs=-1)
            gs.fit(Xp, y_enc)
            model = gs.best_estimator_
            metrics = evaluate_model_cv(Xp, y_enc, estimator=model)
        else:
            svc = SVC(kernel="rbf", probability=True, C=1.0, class_weight="balanced", random_state=42)
            svc.fit(Xp, y_enc)
            model = svc
            metrics = evaluate_model_cv(Xp, y_enc, estimator=model)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "le": le, "metrics": metrics, "scaler": scaler, "pca": pca}, f)
    return model, le, metrics, scaler, pca

def draw_single_face_square(frame, left, right, top, bottom, color, thickness=2):
    h, w = frame.shape[:2]
    face_w = right - left
    face_h = bottom - top
    side = max(face_w, face_h)
    center_x = int((left + right) / 2)
    center_y = int((top + bottom) / 2)
    sq_left = int(center_x - side / 2)
    sq_top = int(center_y - side / 2)
    sq_right = sq_left + side
    sq_bottom = sq_top + side
    sq_left = max(0, min(w - 1, sq_left))
    sq_right = max(0, min(w - 1, sq_right))
    sq_top = max(0, min(h - 1, sq_top))
    sq_bottom = max(0, min(h - 1, sq_bottom))
    if sq_right > sq_left and sq_bottom > sq_top:
        cv2.rectangle(frame, (sq_left, sq_top), (sq_right, sq_bottom), color, thickness)

def enroll_new_face(frame, known_faces):
    name = input("Enter name for new face: ")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        boxes = face_recognition.face_locations(rgb_frame, model="cnn") if USE_CNN_DETECT else face_recognition.face_locations(rgb_frame)
    except Exception:
        boxes = face_recognition.face_locations(rgb_frame)
    try:
        encodings = face_recognition.face_encodings(rgb_frame, boxes)
    except Exception as e:
        print(f"Error during face encoding: {e}")
        return None, None, None, None, None
    if encodings:
        known_faces["encodings"].append(encodings[0])
        known_faces["names"].append(name)
        save_known_faces(known_faces)
        print(f"Enrolled {name}")
        model, le, metrics, scaler, pca = train_classifier(known_faces)
        return model, le, metrics, scaler, pca
    else:
        print("No face detected for enrollment.")
        return None, None, None, None, None

def main():
    known_faces = load_known_faces()
    model, label_enc, metrics, scaler, pca = load_model()
    if model is None and known_faces["encodings"]:
        model, label_enc, metrics, scaler, pca = train_classifier(known_faces)
    if metrics:
        print("Model evaluation metrics:", metrics)
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Press 'e' to enroll a new face, 'q' to quit.")
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            boxes = face_recognition.face_locations(rgb_frame, model="cnn") if USE_CNN_DETECT else face_recognition.face_locations(rgb_frame)
        except Exception:
            boxes = face_recognition.face_locations(rgb_frame)
        if not boxes:
            cv2.imshow("Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                new_model, new_le, new_metrics, new_scaler, new_pca = enroll_new_face(frame, known_faces)
                if new_model is not None:
                    model, label_enc, new_metrics, scaler, pca = new_model, new_le, new_metrics, new_scaler, new_pca
                    if new_metrics:
                        print("Updated model metrics:", new_metrics)
            elif key == ord('q'):
                break
            continue
        try:
            encodings = face_recognition.face_encodings(rgb_frame, boxes)
        except Exception as e:
            print(f"Error during face encoding: {e}")
            cv2.imshow("Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                new_model, new_le, new_metrics, new_scaler, new_pca = enroll_new_face(frame, known_faces)
                if new_model is not None:
                    model, label_enc, new_metrics, scaler, pca = new_model, new_le, new_metrics, new_scaler, new_pca
                    if new_metrics:
                        print("Updated model metrics:", new_metrics)
            elif key == ord('q'):
                break
            continue
        for (top, right, bottom, left), face_encoding in zip(boxes, encodings):
            name = "Unknown"
            color = (0, 0, 255)
            if model is not None and label_enc is not None:
                try:
                    X_in = np.array([face_encoding])
                    if scaler is not None:
                        X_in = scaler.transform(X_in)
                    if pca is not None:
                        X_in = pca.transform(X_in)
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_in)[0]
                        best_index = int(np.argmax(probs))
                        best_prob = float(probs[best_index])
                        enc_label = int(model.classes_[best_index]) if hasattr(model, "classes_") else best_index
                        if best_prob >= PROB_THRESHOLD:
                            name = label_enc.inverse_transform([enc_label])[0]
                            color = (0, 255, 0)
                        else:
                            if known_faces["encodings"]:
                                dists = face_recognition.face_distance(known_faces["encodings"], face_encoding)
                                min_idx = int(np.argmin(dists))
                                if float(dists[min_idx]) <= DIST_THRESHOLD:
                                    name = known_faces["names"][min_idx]
                                    color = (0, 255, 0)
                    else:
                        pred = model.predict(X_in)[0]
                        name = label_enc.inverse_transform([int(pred)])[0]
                        color = (0, 255, 0)
                except Exception:
                    if known_faces["encodings"]:
                        dists = face_recognition.face_distance(known_faces["encodings"], face_encoding)
                        min_idx = int(np.argmin(dists))
                        if float(dists[min_idx]) <= DIST_THRESHOLD:
                            name = known_faces["names"][min_idx]
                            color = (0, 255, 0)
            else:
                if known_faces["encodings"]:
                    dists = face_recognition.face_distance(known_faces["encodings"], face_encoding)
                    min_idx = int(np.argmin(dists))
                    if float(dists[min_idx]) <= DIST_THRESHOLD:
                        name = known_faces["names"][min_idx]
                        color = (0, 255, 0)
            draw_single_face_square(frame, int(left), int(right), int(top), int(bottom), color, thickness=2)
            face_w = int(max(right - left, bottom - top))
            center_x = int((left + right) / 2)
            sq_left = int(center_x - face_w / 2)
            label_x = max(0, sq_left)
            label_y = max(0, int(top) - 10)
            cv2.putText(frame, name, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            new_model, new_le, new_metrics, new_scaler, new_pca = enroll_new_face(frame, known_faces)
            if new_model is not None:
                model, label_enc, metrics, scaler, pca = new_model, new_le, new_metrics, new_scaler, new_pca
                if metrics:
                    print("Updated model metrics:", metrics)
        elif key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
