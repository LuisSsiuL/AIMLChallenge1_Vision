#pragma once

// ============ WiFi ============
#define WIFI_SSID     "AP RITZ ROYAL"
#define WIFI_PASSWORD "Ritzroyal219"

// ============ Camera pins — AI-Thinker ESP32-CAM ============
// Standard AI-Thinker pinout. Do not change unless using different board.
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

// ============ Motors ============
// Bluino ESP32-CAM Motor Shield v3.1 — phase/enable mode (W/A/D working, S not).
#define MOTORS_ENABLED        1

#define LEFT_DRV_PIN          12
#define LEFT_DIR_PIN          14
#define RIGHT_DRV_PIN         13
#define RIGHT_DIR_PIN         15

#define BUZZER_PIN             2    // shield buzzer — keep LOW

#define MOTOR_PWM_FREQ     20000
#define MOTOR_PWM_RES          8
#define MOTOR_SPEED          200
