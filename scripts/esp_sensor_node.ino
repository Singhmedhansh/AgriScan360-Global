/*
  AgriScan360 - ESP Sensor Node Code
  
  Reads temperature & humidity (from a DHT11 or DHT22 sensor) and soil moisture
  (from an analog pin), and sends the telemetry data via HTTP POST in JSON format
  to the deployed FastAPI server.
  
  Requirements in Arduino IDE:
  - Select ESP32 or ESP8266 board.
  - Install "DHT sensor library" by Adafruit.
  - Install "Adafruit Unified Sensor" library.
  - Install "ArduinoJson" library by Benoit Blanchon (optional, but we construct JSON manually here for simplicity to keep it dependency-free).
*/

#if defined(ESP8266)
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#elif defined(ESP32)
#include <WiFi.h>
#include <HTTPClient.h>
#else
#error "This board is not supported. Please use an ESP8266 or ESP32."
#endif

#include <DHT.h>

// ==================== CONFIGURATION ====================
// Wi-Fi Credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Server URL (Replace with your deployed Render URL or local IP)
// Example: "https://agriscan360.onrender.com/sensor_data"
// Or for local testing: "http://192.168.1.100:8000/sensor_data"
const char* serverUrl = "https://YOUR_APP_NAME.onrender.com/sensor_data";

// Device Identifier (to distinguish multiple sensor nodes)
const char* deviceId = "esp_node_01";

// Sensor Pins
#define DHTPIN D4          // Digital pin connected to the DHT sensor (GPIO2 on ESP8266, change as needed for ESP32)
#define DHTTYPE DHT11      // Use DHT11 or DHT22
#define SOIL_PIN A0        // Analog pin connected to soil moisture sensor

// Calibration values for Soil Moisture (adjust based on your sensor readings)
const int AirValue = 1024;   // Sensor value in dry air (0% moisture)
const int WaterValue = 400;  // Sensor value in water (100% moisture)

// Read Interval (in milliseconds)
const unsigned long interval = 30000; // Send reading every 30 seconds
// =======================================================

DHT dht(DHTPIN, DHTTYPE);
unsigned long lastTime = 0;

void setup() {
  Serial.begin(115200);
  delay(10);
  
  dht.begin();
  
  // Connect to Wi-Fi
  Serial.println();
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("Wi-Fi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Wait for the next interval
  if (millis() - lastTime >= interval || lastTime == 0) {
    lastTime = millis();
    
    if (WiFi.status() == WL_CONNECTED) {
      // 1. Read Sensors
      float humidity = dht.readHumidity();
      float temperature = dht.readTemperature(); // Temperature in Celsius
      
      // Check if any reads failed and exit early (to try again).
      if (isnan(humidity) || isnan(temperature)) {
        Serial.println("Failed to read from DHT sensor!");
        return;
      }
      
      // Read Soil Moisture
      int analogValue = analogRead(SOIL_PIN);
      // Map analog reading to a percentage (0% to 100%)
      float soilMoisture = map(analogValue, AirValue, WaterValue, 0, 100);
      // Constrain value between 0% and 100%
      soilMoisture = constrain(soilMoisture, 0, 100);
      
      Serial.println("----------------------------------------");
      Serial.print("Air Temperature: "); Serial.print(temperature); Serial.println(" *C");
      Serial.print("Air Humidity:    "); Serial.print(humidity); Serial.println(" %");
      Serial.print("Soil Moisture:   "); Serial.print(soilMoisture); Serial.print(" % (Raw: "); Serial.print(analogValue); Serial.println(")");

      // 2. Prepare JSON Payload
      // Format: {"device_id": "...", "soil_moisture": ..., "air_temp": ..., "air_humidity": ...}
      String jsonPayload = "{\"device_id\":\"" + String(deviceId) + "\",";
      jsonPayload += "\"soil_moisture\":" + String(soilMoisture, 1) + ",";
      jsonPayload += "\"air_temp\":" + String(temperature, 1) + ",";
      jsonPayload += "\"air_humidity\":" + String(humidity, 1) + "}";
      
      // 3. Send HTTP POST
      WiFiClient client;
      HTTPClient http;
      
      http.begin(client, serverUrl);
      http.addHeader("Content-Type", "application/json");
      
      Serial.print("Sending POST request to: ");
      Serial.println(serverUrl);
      
      int httpResponseCode = http.POST(jsonPayload);
      
      if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.print("HTTP Response code: ");
        Serial.println(httpResponseCode);
        Serial.print("Response payload: ");
        Serial.println(response);
      } else {
        Serial.print("Error code in sending POST: ");
        Serial.println(httpResponseCode);
      }
      
      http.end();
    } else {
      Serial.println("WiFi Disconnected. Reconnecting...");
      WiFi.begin(ssid, password);
    }
  }
}
