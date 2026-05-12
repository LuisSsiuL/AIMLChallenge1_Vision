// ESP32-S3-CAM control node
// - Camera MJPEG stream on  http://<ip>/stream
// - Dashboard UI         on  http://<ip>/
// - Key command          on  http://<ip>/cmd?k=w&s=1   (s=1 down, s=0 up)
//
// Keys: W fwd | S rev | A left | D right | (no key = stop)
// 4 motors as differential drive (left pair + right pair).

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"
#include "config.h"

// Embedded dashboard HTML
extern const char DASHBOARD_HTML[] PROGMEM;

WebServer server(80);

// ============ Motor control ============
// Phase/enable mode (Bluino tutorial). W/A/D confirmed working, S does not.
static bool keyW = false, keyA = false, keyS = false, keyD = false;
static unsigned long lastCmdMs = 0;
static unsigned long cmdCount = 0;
static unsigned long lastWifiLogMs = 0;
static const unsigned long DEADMAN_MS = 1500;  // grace for lossy WiFi (was 600, hot stops)

static void wifiHealthLog() {
  if (millis() - lastWifiLogMs < 3000) return;
  lastWifiLogMs = millis();
  Serial.printf("[wifi] RSSI=%d dBm  cmds/3s=%lu  ip=%s\n",
                WiFi.RSSI(), cmdCount, WiFi.localIP().toString().c_str());
  cmdCount = 0;
}

static void applyMotors() {
#if MOTORS_ENABLED
  bool active = (keyW || keyA || keyS || keyD);

  if (keyW)      { digitalWrite(LEFT_DIR_PIN, HIGH); digitalWrite(RIGHT_DIR_PIN, HIGH); }
  else if (keyS) { digitalWrite(LEFT_DIR_PIN, LOW);  digitalWrite(RIGHT_DIR_PIN, LOW);  }
  else if (keyA) { digitalWrite(LEFT_DIR_PIN, HIGH); digitalWrite(RIGHT_DIR_PIN, LOW);  }
  else if (keyD) { digitalWrite(LEFT_DIR_PIN, LOW);  digitalWrite(RIGHT_DIR_PIN, HIGH); }

  // S barely overcomes friction — give reverse max duty.
  int speed = !active ? 0 : (keyS ? 255 : MOTOR_SPEED);
  ledcWrite(0, speed);
  ledcWrite(4, speed);

  Serial.printf("keys W=%d A=%d S=%d D=%d  L_DIR=%d R_DIR=%d  spd=%d\n",
                keyW, keyA, keyS, keyD,
                digitalRead(LEFT_DIR_PIN), digitalRead(RIGHT_DIR_PIN), speed);
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

// ============ HTTP handlers ============
static void handleRoot() {
  server.send_P(200, "text/html", DASHBOARD_HTML);
}

static void handleCmd() {
  if (!server.hasArg("k") || !server.hasArg("s")) {
    server.send(400, "text/plain", "need k & s");
    return;
  }
  char k = tolower(server.arg("k").charAt(0));
  bool down = server.arg("s") == "1";
  bool changed = false;
  switch (k) {
    case 'w': if (keyW != down) { keyW = down; changed = true; } break;
    case 'a': if (keyA != down) { keyA = down; changed = true; } break;
    case 's': if (keyS != down) { keyS = down; changed = true; } break;
    case 'd': if (keyD != down) { keyD = down; changed = true; } break;
    default: server.send(400, "text/plain", "bad key"); return;
  }
  lastCmdMs = millis();
  cmdCount++;
  if (changed) applyMotors();
  server.send(200, "text/plain", "ok");
}

static void deadmanCheck() {
  if ((keyW || keyA || keyS || keyD) && (millis() - lastCmdMs > DEADMAN_MS)) {
    keyW = keyA = keyS = keyD = false;
    applyMotors();
    Serial.println("[deadman] no cmd, motors stop");
  }
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
    server.handleClient();   // service /cmd etc. between frames
    deadmanCheck();
    wifiHealthLog();
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { delay(10); continue; }
    client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    client.write(fb->buf, fb->len);
    client.print("\r\n");
    esp_camera_fb_return(fb);
    if (!client.connected()) break;
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
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size   = FRAMESIZE_QVGA;  // 320x240
  c.jpeg_quality = 15;              // 0–63 lower=better
  c.fb_count     = 1;
  c.fb_location  = CAMERA_FB_IN_PSRAM;
  c.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("camera init failed 0x%x\n", err);
    return false;
  }
  return true;
}

// ============ Setup / loop ============
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[boot] esp32-cam control node");

  if (!cameraInit()) { Serial.println("halt: camera fail"); while (true) delay(1000); }

  motorsInit();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("[wifi] connecting to %s", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.printf("\n[wifi] ip = http://%s/\n", WiFi.localIP().toString().c_str());
  Serial.printf("[wifi] gateway = %s\n", WiFi.gatewayIP().toString().c_str());
  Serial.printf("[wifi] RSSI = %d dBm  (good > -65, weak < -75)\n", WiFi.RSSI());

  server.on("/", handleRoot);
  server.on("/cmd", handleCmd);
  server.on("/stream", HTTP_GET, handleStream);
  server.begin();
  Serial.println("[http] server up");
}

void loop() {
  server.handleClient();
  deadmanCheck();
  wifiHealthLog();
}
