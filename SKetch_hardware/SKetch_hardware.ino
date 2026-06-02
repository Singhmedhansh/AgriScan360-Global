// ESP8266 ThingSpeak uploader for DHT22 + soil sensor
// Replace the placeholders below before uploading.

#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <DHT.h>

// --- User configuration ---
const char* WIFI_SSID = "Galaxy A53 5G 5984";
const char* WIFI_PASS = "Medhansh";
const char* THINGSPEAK_API_KEY = "ISU09XJXVXHQLRPH"; // write API key
const char* DEVICE_NAME = "ESP8266-AgriKit";

#define DHTPIN D4          // DHT22 sensor
#define DHTTYPE DHT22      // DHT22 sensor
const uint16_t UPLOAD_INTERVAL_MS = 15000; // 15s (ThingSpeak rate limit)

const int SOIL_PIN = A0;   // analog pin for soil sensor
// Calibrate these to your sensor: raw value at dry and wet ends
const int SOIL_RAW_DRY = 800;
const int SOIL_RAW_WET = 300;
const int SOIL_RAW_MISSING_CUTOFF = 20;
// ---------------------------

DHT dht(DHTPIN, DHTTYPE);
unsigned long lastUpload = 0;

void connectWiFi() {
  Serial.printf("Connecting to WiFi '%s'...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(250);
    Serial.print('.');
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected.");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi connect timed out.");
  }
}

String urlEncode(String s) {
  String encoded = "";
  char c;
  char hex[3];
  for (unsigned int i = 0; i < s.length(); i++) {
    c = s.charAt(i);
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      encoded += c;
    } else {
      sprintf(hex, "%02X", c);
      encoded += '%';
      encoded += hex;
    }
  }
  return encoded;
}

void setup() {
  Serial.begin(115200);
  delay(50);
  dht.begin();
  connectWiFi();
}

float readTemperatureC() {
  float t = dht.readTemperature();
  if (isnan(t)) return NAN;
  return t;
}

float readHumidity() {
  float h = dht.readHumidity();
  if (isnan(h)) return NAN;
  return h;
}

int readSoilPercent(int &rawOut) {
  int raw = analogRead(SOIL_PIN);
  rawOut = raw;

  // If reading is out of expected ADC range, treat as missing (-1)
  if (raw < 0 || raw > 1023) return -1;

  // Many disconnected or miswired soil sensors read near zero. Treat that as
  // missing instead of mapping it to 100% moisture.
  if (raw <= SOIL_RAW_MISSING_CUTOFF) return -1;

  int dry = max(SOIL_RAW_DRY, SOIL_RAW_WET);
  int wet = min(SOIL_RAW_DRY, SOIL_RAW_WET);
  raw = constrain(raw, wet, dry);

  // Higher moisture should map to a higher percentage.
  int percent = map(raw, dry, wet, 0, 100);
  return constrain(percent, 0, 100);
}

void sendToThingSpeak(float tempC, float hum, int soilPercent, bool soilPresent) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected, skipping upload.");
    return;
  }

  String url = String("http://api.thingspeak.com/update?api_key=") + THINGSPEAK_API_KEY;
  if (!isnan(tempC)) {
    url += "&field1=" + String(tempC, 2);
  }
  if (!isnan(hum)) {
    url += "&field2=" + String(hum, 2);
  }
  if (soilPresent) {
    url += "&field3=" + String(soilPercent);
  }
  // optional status/device name
  url += "&status=" + urlEncode(String("device: ") + DEVICE_NAME);

  Serial.println("Uploading to ThingSpeak:");
  Serial.println(url);
  WiFiClient client;
  HTTPClient http;
  http.begin(client, url);
  int httpCode = http.GET();
  if (httpCode > 0) {
    String payload = http.getString();
    Serial.printf("ThingSpeak returned: %d / %s\n", httpCode, payload.c_str());
  } else {
    Serial.printf("HTTP error: %d\n", httpCode);
  }
  http.end();
}

void loop() {
  unsigned long now = millis();
  if (now - lastUpload >= UPLOAD_INTERVAL_MS) {
    lastUpload = now;

    float tempC = readTemperatureC();
    float hum = readHumidity();
    int soilRaw = -1;
    int soilPercent = readSoilPercent(soilRaw);
    bool soilPresent = (soilPercent >= 0);

    Serial.print("Temp(C): ");
    if (isnan(tempC)) Serial.print("N/A"); else Serial.print(tempC, 2);
    Serial.print("  Humidity(%): ");
    if (isnan(hum)) Serial.print("N/A"); else Serial.print(hum, 2);
    Serial.print("  Soil(%): ");
    if (soilPresent) Serial.print(soilPercent); else Serial.print("N/A");
    Serial.print("  SoilRaw: ");
    Serial.print(soilRaw);
    Serial.println();

    sendToThingSpeak(tempC, hum, soilPercent, soilPresent);
  }

  // keep WiFi alive; reconnect if dropped
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  delay(200);
}