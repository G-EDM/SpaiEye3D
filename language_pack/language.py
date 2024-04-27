#
# import our stuff
#
from utils.tools import cprint



class dictionary(object):

    def __init__(self,language="de"):

        self.dict = {
            "de":{
                "app_1"   : "App wird beendet"
                ,"app_2"  : "Fertig. Warte auf Bewegungen."
                ,"app_3"  : "Video wird vorbereitet. Dies kann einige Sekunden dauern..."
                ,"app_4"  : "Alle Konturen"
                ,"app_5"  : "Top Kontur"
                ,"app_6"  : "Keine Bewegungen"
                ,"app_7"  : "Total"
                ,"app_8"  : "Zeitstempel"
                ,"app_9"  : "Datum"
                ,"app_10" : "Bild gefunden: "
                ,"app_11" : "Datenbank erfolgreich initialisiert"
                ,"app_12" : "Patrolmodus"
                ,"app_13" : "Videorecorder wird vorbereitet"
                ,"app_14" : "Erstelle Ordner: "
                ,"app_15" : "Für die gewünschte Sprache stehen keine Sprachpakete zur Verfügung"
                ,"app_16" : "Sprachpaket gefunden: "
                ,"app_17" : "Tensorflow Treffer"
                ,"app_18" : "Top Trackingkontur"
                ,"app_19" : "Achtung: Cuda nicht gefunden. CPU Modus aktiviert. Die Geschwindigkeit im CPU Modus ist nicht für Videos geeignet."
                ,"app_20" : "Erstelle Verzeichnis: "
                ,"app_21" : "Speichere Ergebnis unter: "
                ,"app_22" : "Weights erfolgreich geladen: "
                ,"app_23" : "Es wird ein Numpy array erwartet!"
                ,"app_24" : "Bilddatei als Input: "
                ,"app_25" : "Dauer: "
                ,"app_26" : "Videodatei als Input: "
                ,"app_27" : "Das Format der Eingabedatei wird nicht unterstützt: "
                ,"app_28" : "Unterstützte Bildformate: "
                ,"app_29" : "Unterstützte Videoformate: "
                ,"app_30" : "Die Eingabedatei konnte nicht gefunden werden: "
                ,"app_31" : "Verzeichnis nicht gefunden: "
                ,"app_32" : "Bilderverzeichnis: "
                ,"app_33" : "Labelverzeichnis: "
                ,"app_34" : "Speichere Ergebnisse unter: "
                ,"app_35" : "Lade Sprachpaket: "
                ,"app_36" : "Sprachpaket nicht gefunden: "
                ,"app_37" : "Video- oder Bilddatei zum analysieren"
                ,"app_38" : "Speichern der Videos und Bilder mit Boxen; (True/False); Default:True"
                ,"app_38" : "Speichern der Videos und Bilder mit Boxen; (True/False); Default:True"
                ,"app_39" : "Ergebnis in einem Fenster anzeigen; Default:True"
                ,"app_40" : "YOLOv3 Weightsdatei wird geladen. Je nach Dateigröße kann dies etwas dauern."
                ,"app_41" : "DeepSort Checkpoint wird geladen. Je nach Dateigröße kann dies etwas dauern."
                ,"app_42" : "Pfad zum Ordner mit den Labeldateien im .txt Format."
                ,"app_43" : "Die Klasse zum entfernen. (int)"
                ,"app_44" : "Durchsuche den Ordner: "
                ,"app_45" : "Initialisiert"
                ,"app_46" : "Überschreibe Labelsdatei: "
                ,"app_47" : "Duplikat in: "
                ,"app_48" : "Fehler: "
                ,"app_49" : "Gast"
                ,"app_50" : "Bilderordner zum labeln"
                ,"app_51" : "Speichern"
                ,"app_52" : "Speichere Datei: "
                ,"app_53" : "Videostream wird aufgenommen"
                ,"app_54" : "Monitor mode. Bewegungen werden aufgezeichnet"
                ,"app_55" : "Aktuellen Frame speichern"
                ,"app_56" : "Konfigurationsdatei wird geladen"
                ,"app_57" : "Konfigurationsdatei konnte nicht geladen werden"
                ,"app_58" : "Serverconfig hat sich geändert"
                ,"app_59" : "Verbindung zum Server fehlgeschlagen"
                ,"app_60" : "Verbindung zum Server hergestellt"
                ,"app_61" : "Starte Frameemitter"
                ,"app_62" : "Stoppe Frameemitter"
                ,"app_63" : "Paketgröße: "
                ,"app_64" : "Schlüssel akzeptiert: "
                ,"app_65" : "Schlüssel nicht akzeptiert: "
                ,"app_66" : "Eingehende Verbindung"
                ,"app_67" : "Server lauscht an Port: "
                ,"app_68" : "Versuche: "
                ,"app_69" : "run_stream(): "
                ,"app_70" : "Recorderpipe bereit"
                ,"app_71" : "Warte auf eingehende Verbindung"
                ,"app_72" : "SpaiEye3D"
                ,"app_73" : "Einen Augenblick..."
                ,"app_74" : "WebcamModus"
                ,"app_75" : "RemoteWebcamModus"
                ,"app_76" : "Darstellung wird neu berechnet"
                ,"app_77" : "Buttons werden neu berechnet"
                ,"app_78" : "Frames pro Sekunde"
                ,"app_79" : "Reduziere FPS"
                ,"app_80" : "Erzeuge Startbildschirm"
                ,"app_81" : "Fenstergröße wird neu berechnet"
                ,"app_82" : "Starte Frameemitter"
                ,"app_83" : "Nach oben bewegen"
                ,"app_84" : "Nach unten bewegen"
                ,"app_85" : "Nach links bewegen"
                ,"app_86" : "Nach rechts bewegen"
                ,"app_87" : "Netz an/aus"
                ,"app_88" : "Extractionmode"
                ,"app_89" : "Webcam nicht gefunden"
            }
        }

        self.success  = True
        self.language = language.lower()
        self.language = self.language.strip()

        if self.language not in self.dict:
            self.success  = False
            self.language = "de"



    def print_result(self):
        if not self.success:
            cprint( self.get("app_36")+self.language,"WARNING" )
        cprint( self.get("app_35")+self.language )


    def get(self,item):
        if type(item) == int:
            item = "app_"+str(item)
        return self.dict[self.language][item]


