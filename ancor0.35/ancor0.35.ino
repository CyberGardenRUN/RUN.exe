#include <BLEDevice.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>

#define TARGET_NAME "TAG_123"
#define ANCHOR_ID   "A0"      // УНИКАЛЬНО для каждого якоря!
#define SCAN_INTERVAL_MS 120
#define SCAN_WINDOW_MS   120
#define SCAN_DURATION_S  0

class AdvCallbacks : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice d) override {
    if (!d.haveName()) return;
    String name = d.getName();
    if (name != TARGET_NAME) return;
    int rssi = d.getRSSI();
    // Печатаем ТОЛЬКО сырой RSSI + id якоря
    Serial.printf("ANCHOR:%s RSSI:%d\n", ANCHOR_ID, rssi);
  }
};

void setup() {
  Serial.begin(115200);
  BLEDevice::init(String("ANCHOR_") + ANCHOR_ID);
  BLEDevice::setPower(ESP_PWR_LVL_P9);

  BLEScan* scan = BLEDevice::getScan();
  scan->setAdvertisedDeviceCallbacks(new AdvCallbacks(), true);
  scan->setActiveScan(true);
  scan->setInterval((uint16_t)(SCAN_INTERVAL_MS / 0.625));
  scan->setWindow((uint16_t)(SCAN_WINDOW_MS   / 0.625));
  scan->start(SCAN_DURATION_S, false);

  Serial.printf("READY %s\n", ANCHOR_ID);
}

void loop(){ delay(10); }
