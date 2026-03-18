/*
  Seedling Tracker - ESP32-WROVER-CAM
  
  Serves JPEG snapshots over HTTP at /capture
  Designed for overhead hydroponic seedling monitoring.
  
  Endpoints:
    GET /capture   - returns a JPEG snapshot
    GET /status    - returns JSON with uptime and settings
    GET /stream    - MJPEG stream for live preview (use sparingly)
  
  Adjust CAMERA_MODEL, WiFi credentials, and resolution below.
*/

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "fb_gfx.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ==================== CONFIG ====================
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Resolution: higher = more detail but slower transfer
// FRAMESIZE_UXGA  = 1600x1200  (may be unstable on some boards)
// FRAMESIZE_SXGA  = 1280x1024
// FRAMESIZE_XGA   = 1024x768   (recommended balance)
// FRAMESIZE_SVGA  = 800x600
#define CAPTURE_RESOLUTION FRAMESIZE_XGA

// JPEG quality: 10 (best) to 63 (worst), 12-15 is a good range
#define JPEG_QUALITY 12

// ==================== CAMERA PINS (ESP32-WROVER-CAM / ESP32-CAM AI-Thinker) ====================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Onboard flash LED (GPIO 4 on most ESP32-CAM boards)
#define FLASH_LED_PIN      4

WiFiServer server(80);
unsigned long bootTime;

void setup() {
  Serial.begin(115200);
  Serial.println("\n\nSeedling Tracker Camera Starting...");
  
  // Disable brownout detector (ESP32-CAM draws spikes on capture)
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  
  // Flash LED off
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);
  
  // ---- Camera config ----
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // WROVER has PSRAM — use it for higher resolution
  if (psramFound()) {
    Serial.println("PSRAM found! Using high-res config.");
    config.frame_size   = CAPTURE_RESOLUTION;
    config.jpeg_quality = JPEG_QUALITY;
    config.fb_count     = 2;  // double buffer for reliability
    config.grab_mode    = CAMERA_GRAB_LATEST;
  } else {
    Serial.println("No PSRAM — falling back to lower res.");
    config.frame_size   = FRAMESIZE_SVGA;
    config.jpeg_quality = 15;
    config.fb_count     = 1;
  }
  
  // Init camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    ESP.restart();
  }
  
  // Lock white balance and exposure for consistency
  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    s->set_whitebal(s, 1);       // enable white balance
    s->set_awb_gain(s, 1);       // enable AWB gain
    s->set_wb_mode(s, 0);        // 0=auto, 1=sunny, 2=cloudy, 3=office, 4=home
    // After initial auto-adjust, you may want to lock:
    // s->set_whitebal(s, 0);    // lock WB after warmup
    // s->set_exposure_ctrl(s, 0); // lock exposure
    s->set_brightness(s, 0);     // -2 to 2
    s->set_contrast(s, 0);       // -2 to 2
    s->set_saturation(s, 0);     // -2 to 2
  }
  
  // ---- WiFi connect ----
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);  // keep WiFi alive for reliable captures
  
  Serial.print("Connecting to WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi failed! Restarting...");
    ESP.restart();
  }
  
  Serial.println("\nWiFi connected!");
  Serial.print("Camera ready at: http://");
  Serial.println(WiFi.localIP());
  
  server.begin();
  bootTime = millis();
}

void handleCapture(WiFiClient &client) {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    client.println("HTTP/1.1 500 Internal Server Error");
    client.println("Content-Type: text/plain");
    client.println();
    client.println("Camera capture failed");
    return;
  }
  
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: image/jpeg");
  client.printf("Content-Length: %d\r\n", fb->len);
  client.println("Access-Control-Allow-Origin: *");
  client.println("Connection: close");
  client.println();
  
  client.write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

void handleStatus(WiFiClient &client) {
  unsigned long uptime = (millis() - bootTime) / 1000;
  
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: application/json");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Connection: close");
  client.println();
  client.printf("{\"status\":\"ok\",\"uptime_s\":%lu,\"ip\":\"%s\",\"psram\":%s,\"rssi\":%d}",
    uptime,
    WiFi.localIP().toString().c_str(),
    psramFound() ? "true" : "false",
    WiFi.RSSI()
  );
}

void handleStream(WiFiClient &client) {
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println("Access-Control-Allow-Origin: *");
  client.println();
  
  while (client.connected()) {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) continue;
    
    client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", fb->len);
    client.write(fb->buf, fb->len);
    client.println();
    esp_camera_fb_return(fb);
    
    delay(100);  // ~10 fps max for preview
  }
}

void loop() {
  WiFiClient client = server.available();
  if (!client) return;
  
  String request = "";
  unsigned long timeout = millis() + 3000;
  while (client.connected() && millis() < timeout) {
    if (client.available()) {
      char c = client.read();
      request += c;
      if (request.endsWith("\r\n\r\n")) break;
    }
  }
  
  if (request.indexOf("GET /capture") >= 0) {
    handleCapture(client);
  } else if (request.indexOf("GET /status") >= 0) {
    handleStatus(client);
  } else if (request.indexOf("GET /stream") >= 0) {
    handleStream(client);
  } else {
    // Default: simple info page
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: text/html");
    client.println();
    client.println("<html><body>");
    client.println("<h2>Seedling Tracker Camera</h2>");
    client.println("<p><a href='/capture'>Capture JPEG</a></p>");
    client.println("<p><a href='/stream'>Live Stream</a></p>");
    client.println("<p><a href='/status'>Status JSON</a></p>");
    client.println("</body></html>");
  }
  
  client.stop();
}
