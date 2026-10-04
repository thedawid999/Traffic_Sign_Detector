# 🚦 Traffic Sign Detector
***
## 👤 Projektinformationen

| **Autor** | thedawid999 |
| :--- | :--- |
| **Studiengang** | Angewandte Künstliche Intelligenz |
| **Projekt/Modul** | Computer Vision |

***

## 🌟 Projektziel

Die Hauptanforderung dieses Projekts ist die Entwicklung und umfassende Evaluation eines **Deep-Learning-basierten Modells** zur **Echtzeit-Verkehrszeichenerkennung**. Das Ziel ist es, eine hohe Zuverlässigkeit und Robustheit unter variablen Bedingungen (Licht, Entfernung) bei gleichzeitiger Erreichung einer niedrigen Latenz (mindestens **30 FPS**) für den Einsatz in Fahrerassistenzsystemen zu gewährleisten.

Um den optimalen Kompromiss zu finden, wurden zwei Ansätze verglichen:
1.  **Ansatz A:** YOLO + CNN (Kombinierter Detektions- und Klassifikationsansatz)
2.  **Ansatz B:** YOLO-only (Monolithischer Single-Stage-Detektor)

***

## 🛠️ Architektur und Methodik

### Ansatz A: YOLO + CNN (Kombinierte Lösung)

Dieser zweigleisige Ansatz trennt Lokalisierung und Klassifikation, um die Gesamtleistung zu steigern.

* **1. Detektion (YOLOv11n):** Ein YOLO-Modell ist für die Lokalisierung und das Ausschneiden der Bounding Boxes für die generische Klasse "Verkehrszeichen" verantwortlich.
* **2. Klassifikation (Eigenes CNN):** Ein separates, selbst entwickeltes CNN klassifiziert den ausgeschnittenen Bildausschnitt präzise in eine der **43 Verkehrszeichen-Klassen**.

### Ansatz B: YOLO-only (Single-Stage)

Dieser monolithische Detektor führt Objektdetektion und Klassifizierung in einem einzigen Durchlauf durch.

* **Modell:** Die leistungsstärkere **YOLOv11s**-Variante wurde direkt auf den Datensätzen für Detektion und Klassifikation trainiert.

***

## 📚 Verwendete Technologien

Das Projekt basiert auf der Programmiersprache **Python** und den folgenden Schlüsselbibliotheken:

| Technologie | Rolle im Projekt |
| :--- | :--- |
| **Ultralytics** | Training und Evaluation der **YOLO**-Modelle (YOLOv11n/s). |
| **TensorFlow/Keras** | Erstellung und Training des separaten **eigenen CNNs**. |
| **OpenCV** | **Echtzeit-Bild- und Videoanalyse**, Darstellung der Bounding Boxes. |
| **NumPy** | Effiziente Berechnung mit Bilddaten. |

***

## 💾 Datengrundlage

Für das Training und die Evaluation wurden zwei etablierte deutsche Benchmarks verwendet:

| Datensatz | Fokus | # Klassen | Zweck |
| :--- | :--- | :--- | :--- |
| **GTSDB** | **Lokalisierung** | 1 (VKZ allgemein) | Erkennung der Position von Verkehrszeichen in unzugeschnittenen Bildern (ca. 900 Bilder). |
| **GTSRB** | **Klassifizierung** | 43 | Training der präzisen Klassifikation der 43 unterschiedlichen Verkehrszeichen (über 50.000 Bilder). |

***

## 🚀 Installation und Ausführung

### 1. Abhängigkeiten installieren

Installieren Sie die notwendigen Bibliotheken in Ihrer Python-Umgebung:

```bash
ultralytics==8.3.203
tensorflow==2.10.0
keras==2.10.0
opencv-python==4.7.0.72
numpy==1.24.2
scikit-learn==1.7.2
matplotlib==3.10.6
```

### 2. Ausführen

Wählen Sie eine oder mehrere Methoden (`yolo_picutre()`, `yolo_live()`, `picutre()`, `live()`) in der `main.py` und führen Sie diese aus.

## 🎯 Ergebnisse

Die Ergebnisse dieses Projekts sind ebenfalls im **outputs** Ordner zu finden

| YOLO + CNN | YOLO only |
|:---:|:---:|
| <img width="1360" height="800" alt="0_detected_yolo_cnn" src="https://github.com/user-attachments/assets/1b8ad479-8b38-42c0-b883-5bc04f454c37" />  | <img width="1360" height="800" alt="0_detected_yolo_only" src="https://github.com/user-attachments/assets/45cd9aa0-46ba-40a6-8202-ca5dae94c001" /> |
| <img width="1360" height="800" alt="1_detected_yolo_cnn" src="https://github.com/user-attachments/assets/eb450a40-b5c1-42a4-9a4a-5f4f67dcf1c0" /> | <img width="1360" height="800" alt="1_detected_yolo_only" src="https://github.com/user-attachments/assets/d6bb71af-9a7a-45ab-9d03-ee7a0276a8a2" /> | 
| <img width="1360" height="800" alt="3_detected_yolo_cnn" src="https://github.com/user-attachments/assets/d20dfed0-5595-44e1-87c7-ef4ee92b8eb9" /> | <img width="1360" height="800" alt="3_detected_yolo_only" src="https://github.com/user-attachments/assets/c36f268d-5041-48d3-ae9a-497ddf5beed1" /> |
| <img width="1307" height="262" alt="combined_detected_yolo_cnn" src="https://github.com/user-attachments/assets/f0cdc470-39d4-4a0f-903d-5ac71e72139f" /> | <img width="1307" height="262" alt="combined_detected_yolo_only" src="https://github.com/user-attachments/assets/8201cac5-cf57-4f82-bd8a-8a1ddf2c0e38" /> |
| <img width="1200" height="500" alt="latency_fps_yolo_cnn" src="https://github.com/user-attachments/assets/8a71e3cc-5f5a-40a9-9f4a-e746f3587190" /> | <img width="1200" height="500" alt="latency_fps_yolo_only" src="https://github.com/user-attachments/assets/760a125f-eb11-4264-8165-b3d78284cdb6" /> |
| <img width="1904" height="969" alt="Screenshot 2026-10-04 163456" src="https://github.com/user-attachments/assets/638c45bd-cc8c-42ae-83ad-16e9923d96e3" /> | <img width="1909" height="936" alt="Screenshot 2026-10-04 163439" src="https://github.com/user-attachments/assets/ef8b7bc4-c793-463b-ab21-de0402e0b22a" /> |


