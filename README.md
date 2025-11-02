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

Die Ergebnisse dieses Projekts sind im **outputs** Ordner zu finden
