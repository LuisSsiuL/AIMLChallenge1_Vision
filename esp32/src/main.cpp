// ESP32-CAM control node
// - Camera MJPEG stream on  http://<ip>/stream
// - Dashboard UI         on  http://<ip>/
// - WebSocket cmd        on  ws://<ip>:81/   (msg: "w1" / "w0" / "a1" / ...)
// - HTTP cmd fallback    on  http://<ip>/cmd?k=w&s=1
// - OTA firmware upload  on  http://<ip>/update
//
// Keys: W fwd | S rev | A left | D right | (no key = stop)
// 4 motors as differential drive (left pair + right pair).
// WIFI_AP_MODE in config.h toggles AP <-> STA.

#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <ESPmDNS.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <ElegantOTA.h>
#include "esp_camera.h"
#include "config.h"

// mDNS hostname — Swift resolves "espcar.local" so Config.swift never
// needs the DHCP-assigned IP. Works in both STA and AP modes (responder
// runs on whichever interface WiFi.mode() activated).
#ifndef MDNS_HOSTNAME
#define MDNS_HOSTNAME "espcar"
#endif

extern const char DASHBOARD_HTML[] PROGMEM;

WebServer server(80);
WebSocketsServer ws(81);

// ============ Live stats ============
static volatile uint32_t g_frames = 0;
static volatile uint32_t g_bytes  = 0;
static volatile float    g_fps    = 0;
static volatile uint32_t g_kbps   = 0;
static uint32_t          g_lastStatsMs = 0;

// ============ Motor control ============
static bool keyW = false, keyA = false, keyS = false, keyD = false;

static void applyMotors() {
#if MOTORS_ENABLED
  // Differential drive:
  //   W      : both fwd full
  //   S      : both rev full
  //   A      : tank-turn left  (L rev, R fwd)
  //   D      : tank-turn right (L fwd, R rev)
  //   W+A    : arc fwd-left  (L slow fwd,  R full fwd)
  //   W+D    : arc fwd-right (L full fwd,  R slow fwd)
  //   S+A    : arc rev-left  (L slow rev,  R full rev)
  //   S+D    : arc rev-rev-right
  int leftDir = LOW, rightDir = LOW;
  int leftSpd = 0,   rightSpd = 0;

  if (keyW && keyA)      { leftDir = HIGH; rightDir = HIGH; leftSpd = MOTOR_SPEED;       rightSpd = MOTOR_SPEED_INNER; }
  else if (keyW && keyD) { leftDir = HIGH; rightDir = HIGH; leftSpd = MOTOR_SPEED_INNER; rightSpd = MOTOR_SPEED;       }
  else if (keyS && keyA) { leftDir = LOW;  rightDir = LOW;  leftSpd = MOTOR_SPEED;       rightSpd = MOTOR_SPEED_INNER; }
  else if (keyS && keyD) { leftDir = LOW;  rightDir = LOW;  leftSpd = MOTOR_SPEED_INNER; rightSpd = MOTOR_SPEED;       }
  else if (keyW)         { leftDir = HIGH; rightDir = HIGH; leftSpd = MOTOR_SPEED;       rightSpd = MOTOR_SPEED;       }
  else if (keyS)         { leftDir = LOW;  rightDir = LOW;  leftSpd = MOTOR_SPEED;       rightSpd = MOTOR_SPEED;       }
  else if (keyA)         { leftDir = HIGH; rightDir = LOW;  leftSpd = MOTOR_SPEED;       rightSpd = MOTOR_SPEED;       }
  else if (keyD)         { leftDir = LOW;  rightDir = HIGH; leftSpd = MOTOR_SPEED;       rightSpd = MOTOR_SPEED;       }

  digitalWrite(LEFT_DIR_PIN,  leftDir);
  digitalWrite(RIGHT_DIR_PIN, rightDir);
  ledcWrite(0, leftSpd);
  ledcWrite(4, rightSpd);

  Serial.printf("keys W=%d A=%d S=%d D=%d  L_DIR=%d R_DIR=%d  L_spd=%d R_spd=%d\n",
                keyW, keyA, keyS, keyD,
                leftDir, rightDir, leftSpd, rightSpd);
#endif
}

static void motorsInit() {
#if MOTORS_ENABLED
  pinMode(BUZZER_PIN,    OUTPUT); digitalWrite(BUZZER_PIN,    LOW);
  pinMode(LEFT_DIR_PIN,  OUTPUT); digitalWrite(LEFT_DIR_PIN,  LOW);
  pinMode(RIGHT_DIR_PIN, OUTPUT); digitalWrite(RIGHT_DIR_PIN, LOW);
  ledcSetup(0, MOTOR_PWM_FREQ, MOTOR_PWM_RES); ledcAttachPin(LEFT_DRV_PIN,  0);
  ledcSetup(4, MOTOR_PWM_FREQ, MOTOR_PWM_RES); ledcAttachPin(RIGHT_DRV_PIN, 4);
  applyMotors();
#endif
}

// Parse "<key><state>" e.g. "w1" / "s0". Returns true if applied.
static bool applyKeyMsg(const char *msg, size_t len) {
  if (len < 2) return false;
  char k = tolower(msg[0]);
  bool down = msg[1] == '1';
  switch (k) {
    case 'w': keyW = down; break;
    case 'a': keyA = down; break;
    case 's': keyS = down; break;
    case 'd': keyD = down; break;
    default: return false;
  }
  applyMotors();
  return true;
}

// ============ HTTP handlers ============
static void handleRoot() {
  server.send_P(200, "text/html", DASHBOARD_HTML);
}

// Legacy HTTP cmd — kept for curl/test; dashboard uses WS now.
static void handleCmd() {
  if (!server.hasArg("k") || !server.hasArg("s")) {
    server.send(400, "text/plain", "need k & s");
    return;
  }
  String msg = server.arg("k") + server.arg("s");
  if (!applyKeyMsg(msg.c_str(), msg.length())) {
    server.send(400, "text/plain", "bad key");
    return;
  }
  server.send(200, "text/plain", "ok");
}

// MJPEG streaming — pulls JPEG from camera, wraps in multipart
static void handleStream() {
  WiFiClient client = server.client();
  String hdr =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
    "Access-Control-Allow-Origin: *\r\n\r\n";
  client.print(hdr);

  while (client.connected()) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { delay(10); continue; }
    client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    client.write(fb->buf, fb->len);
    client.print("\r\n");
    g_frames++;
    g_bytes += fb->len;
    esp_camera_fb_return(fb);
    if (!client.connected()) break;
  }
}

// Pinned to core 0 — keeps WS cmd path independent of frame send on core 1.
static void wsTask(void *) {
  for (;;) {
    ws.loop();
    vTaskDelay(1 / portTICK_PERIOD_MS);
  }
}

static void handleStats() {
  uint32_t now = millis();
  uint32_t dt  = now - g_lastStatsMs;
  if (dt >= 1000) {
    g_fps  = (g_frames * 1000.0f) / dt;
    g_kbps = (g_bytes  * 8) / dt;   // bytes/ms = kbits/s
    g_frames = 0;
    g_bytes  = 0;
    g_lastStatsMs = now;
  }
  String json = "{";
  json += "\"rssi\":" + String(WiFi.RSSI()) + ",";
  json += "\"fps\":" + String(g_fps, 1) + ",";
  json += "\"kbps\":" + String(g_kbps) + ",";
  json += "\"heap\":" + String(ESP.getFreeHeap()) + ",";
  json += "\"psram\":" + String(ESP.getFreePsram());
  json += "}";
  server.send(200, "application/json", json);
}

// ============ WebSocket ============
static void wsEvent(uint8_t num, WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED: {
      IPAddress ip = ws.remoteIP(num);
      Serial.printf("[ws] %u connected from %s\n", num, ip.toString().c_str());
      break;
    }
    case WStype_DISCONNECTED:
      Serial.printf("[ws] %u disconnected — clearing keys\n", num);
      // Safety: drop all keys if a client disconnects mid-press.
      keyW = keyA = keyS = keyD = false;
      applyMotors();
      break;
    case WStype_TEXT:
      applyKeyMsg((const char *)payload, length);
      break;
    default: break;
  }
}

// ============ Camera ============
static bool cameraInit() {
  camera_config_t c{};
  c.ledc_channel = LEDC_CHANNEL_2;   // avoid 0/1 used by motors
  c.ledc_timer   = LEDC_TIMER_1;
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 16000000;        // 16MHz: less WiFi interference vs 20MHz on classic ESP32
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size   = FRAMESIZE_QVGA;  // start small for low latency; /res switches up
  c.jpeg_quality = 15;              // 0–63 lower=better
  c.fb_count     = 2;               // double-buffer: capture next while sending current
  c.fb_location  = CAMERA_FB_IN_PSRAM;
  c.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("camera init failed 0x%x\n", err);
    return false;
  }

  // Runtime sensor tuning — image looks dull/dark out of the box.
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 1);     // -2..2
    s->set_contrast(s, 1);       // -2..2
    s->set_saturation(s, 1);     // -2..2
    s->set_whitebal(s, 1);       // auto white balance
    s->set_awb_gain(s, 1);       // AWB gain
    s->set_wb_mode(s, 0);        // 0=auto, 1=sunny, 2=cloudy, 3=office, 4=home
    s->set_exposure_ctrl(s, 1);  // auto exposure
    s->set_aec2(s, 1);           // AEC DSP
    s->set_gain_ctrl(s, 1);      // auto gain
    s->set_lenc(s, 1);           // lens correction
    s->set_hmirror(s, 0);
    s->set_vflip(s, 0);
  }
  return true;
}

// Runtime resolution switch: /res?size=qvga|vga|svga|hd|uxga
static framesize_t parseSize(const String &name) {
  if (name == "qqvga") return FRAMESIZE_QQVGA;  // 160x120
  if (name == "qvga")  return FRAMESIZE_QVGA;   // 320x240
  if (name == "cif")   return FRAMESIZE_CIF;    // 400x296
  if (name == "vga")   return FRAMESIZE_VGA;    // 640x480
  if (name == "svga")  return FRAMESIZE_SVGA;   // 800x600
  if (name == "xga")   return FRAMESIZE_XGA;    // 1024x768
  if (name == "hd")    return FRAMESIZE_HD;     // 1280x720
  if (name == "sxga")  return FRAMESIZE_SXGA;   // 1280x1024
  if (name == "uxga")  return FRAMESIZE_UXGA;   // 1600x1200
  return FRAMESIZE_INVALID;
}

static void handleRes() {
  if (!server.hasArg("size")) {
    server.send(400, "text/plain", "need ?size=qvga|vga|svga|hd|uxga");
    return;
  }
  framesize_t fs = parseSize(server.arg("size"));
  if (fs == FRAMESIZE_INVALID) { server.send(400, "text/plain", "bad size"); return; }
  sensor_t *s = esp_camera_sensor_get();
  if (!s) { server.send(500, "text/plain", "no sensor"); return; }
  int rc = s->set_framesize(s, fs);
  // Optional quality tweak: higher res → loosen quality to avoid corruption/crash.
  if (server.hasArg("q")) s->set_quality(s, server.arg("q").toInt());
  server.send(200, "text/plain", rc == 0 ? "ok" : "set_framesize failed");
}

// ============ WiFi ============
static void wifiInit() {
#if WIFI_AP_MODE
  WiFi.mode(WIFI_AP);
  // Channel 6 — less crowded than default 1 in most apartments.
  // ssid_hidden=0, max_connection=1 (only the Mac), beacon every 100ms.
  WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASS, /*channel*/ 6, /*hidden*/ 0, /*maxConn*/ 1);
  // Disable AP power-save so beacons + frames are sent immediately.
  esp_wifi_set_ps(WIFI_PS_NONE);
  // Max TX power (8.5 dBm units; 78 = 19.5 dBm = max for ESP32).
  esp_wifi_set_max_tx_power(78);
  IPAddress ip = WiFi.softAPIP();
  Serial.printf("[wifi] AP up — ssid=%s pass=%s ch=6 ip=http://%s/\n",
                WIFI_AP_SSID, WIFI_AP_PASS, ip.toString().c_str());
#else
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);              // disable PS — eliminates 100–300ms stalls
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("[wifi] joining %s", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.printf("\n[wifi] ip = http://%s/  rssi=%d dBm\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI());
#endif

  // mDNS responder — Swift connects to "espcar.local" instead of the
  // DHCP-assigned IP. Advertise HTTP (camera/dashboard) and WS service
  // records so Bonjour browsers can discover the node too.
  if (MDNS.begin(MDNS_HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    MDNS.addService("ws",   "tcp", 81);
    Serial.printf("[mdns] hostname = %s.local\n", MDNS_HOSTNAME);
  } else {
    Serial.println("[mdns] start failed");
  }
}

// ============ Setup / loop ============
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[boot] esp32-cam control node");

  if (!cameraInit()) { Serial.println("halt: camera fail"); while (true) delay(1000); }

  motorsInit();
  wifiInit();

  server.on("/", handleRoot);
  server.on("/cmd", handleCmd);
  server.on("/res", handleRes);
  server.on("/stats", handleStats);
  server.on("/stream", HTTP_GET, handleStream);

  ElegantOTA.begin(&server);   // mounts /update
  server.begin();

  ws.begin();
  ws.onEvent(wsEvent);
  // Run WS loop on core 0 — frame sending on core 1 (Arduino loop) won't starve cmd path.
  xTaskCreatePinnedToCore(wsTask, "wsTask", 4096, nullptr, 2, nullptr, 0);

  Serial.println("[http] server up — / dashboard, /stream, /cmd, /update");
  Serial.println("[ws]   server up on :81");
}

void loop() {
  server.handleClient();
  ElegantOTA.loop();
  // ws.loop() runs in its own task on core 0 — see wsTask.
}
