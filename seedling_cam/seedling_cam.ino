/*
  Seedling Tracker - ESP32-WROVER-CAM
  
  Serves JPEG snapshots over HTTP for the Mac Mini tracker.
  Uses ESPAsyncWebServer (same stack as proven plant-camera firmware).
  
  Endpoints:
    GET /capture   - returns a JPEG snapshot
    GET /status    - returns JSON with uptime and settings
    GET /stream    - MJPEG stream for live preview
*/

#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <esp_camera.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ==================== CONFIG ====================
const char* ssid     = "222_V768";
const char* password = "5875b81f";

#define CAPTURE_RESOLUTION FRAMESIZE_XGA  // 1024x768
#define JPEG_QUALITY 12

// ==================== CAMERA PINS (ESP32-WROVER / Freenove) ====================
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    21
#define SIOD_GPIO_NUM    26
#define SIOC_GPIO_NUM    27
#define Y9_GPIO_NUM      35
#define Y8_GPIO_NUM      34
#define Y7_GPIO_NUM      39
#define Y6_GPIO_NUM      36
#define Y5_GPIO_NUM      19
#define Y4_GPIO_NUM      18
#define Y3_GPIO_NUM       5
#define Y2_GPIO_NUM       4
#define VSYNC_GPIO_NUM   25
#define HREF_GPIO_NUM    23
#define PCLK_GPIO_NUM    22

// ==================== GLOBALS ====================
AsyncWebServer server(80);
unsigned long bootTime;

void setup() {
  Serial.begin(115200);
  Serial.println("\n\nSeedling Tracker Camera Starting...");
  
  // Disable brownout detector
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  
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
  
  if (psramFound()) {
    Serial.println("PSRAM found! Using high-res config.");
    config.frame_size   = CAPTURE_RESOLUTION;
    config.jpeg_quality = JPEG_QUALITY;
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;
  } else {
    Serial.println("No PSRAM — falling back to lower res.");
    config.frame_size   = FRAMESIZE_SVGA;
    config.jpeg_quality = 15;
    config.fb_count     = 1;
  }
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    ESP.restart();
  }
  
  // Sensor settings
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, 1);       // flip for overhead mount
    s->set_hmirror(s, 1);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 0);     // auto
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
  }
  
  // ---- WiFi connect ----
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);
  
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
  
  bootTime = millis();
  
  // ---- Routes ----
  server.on("/capture", HTTP_GET, [](AsyncWebServerRequest *request) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      request->send(500, "text/plain", "Camera capture failed");
      return;
    }
    AsyncWebServerResponse *response = request->beginResponse_P(200, "image/jpeg", fb->buf, fb->len);
    response->addHeader("Access-Control-Allow-Origin", "*");
    request->send(response);
    esp_camera_fb_return(fb);
  });
  
  server.on("/status", HTTP_GET, [](AsyncWebServerRequest *request) {
    unsigned long uptime = (millis() - bootTime) / 1000;
    char json[256];
    snprintf(json, sizeof(json),
      "{\"status\":\"ok\",\"uptime_s\":%lu,\"ip\":\"%s\",\"psram\":%s,\"rssi\":%d}",
      uptime,
      WiFi.localIP().toString().c_str(),
      psramFound() ? "true" : "false",
      WiFi.RSSI()
    );
    AsyncWebServerResponse *response = request->beginResponse(200, "application/json", json);
    response->addHeader("Access-Control-Allow-Origin", "*");
    request->send(response);
  });
  
  // Simple info page
  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request) {
    request->send(200, "text/html",
      "<html><body>"
      "<h2>Seedling Tracker Camera</h2>"
      "<p><a href='/capture'>Capture JPEG</a></p>"
      "<p><a href='/stream'>Live Stream</a></p>"
      "<p><a href='/status'>Status JSON</a></p>"
      "</body></html>"
    );
  });
  
  server.begin();
  Serial.println("HTTP server started.");
}

void loop() {
  // Nothing needed — async server handles requests
  delay(1000);
}
