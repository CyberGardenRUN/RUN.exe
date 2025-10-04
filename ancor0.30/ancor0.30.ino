#include <BLEDevice.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>
#include <math.h>

#define TARGET_NAME       "TAG_123"   // имя BLE-метки (как в коде TAG)
#define SCAN_INTERVAL_MS  120         // период сканирования
#define SCAN_WINDOW_MS    120         // окно = период => непрерывный активный скан
#define SCAN_DURATION_S   0           // 0 = непрерывно

// --- модель расстояния по RSSI ---
// d = 10^((TxPower_1m - RSSI)/(10*n))
// где n ≈ 1.7..2.2 для помещений; TxPower_1m — калибруем.
// Начни с этих, потом подстрой по факту.
float N_PLF       = 2.0f;   // показатель затухания среды
int   TXPWR_1M_DB = -59;    // ожидаемый RSSI на 1 м (калибруется)

// сглаживание RSSI (EMA)
float rssiEMA = NAN;
const float ALPHA = 0.3f;   // 0.2..0.5 — компромисс шум/задержка

class AdvCallbacks : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice advertisedDevice) override {
    if (!advertisedDevice.haveName()) return;
    String name = advertisedDevice.getName();  // Используем тип String
    if (name != TARGET_NAME) return;

    int rssi = advertisedDevice.getRSSI();

    // EMA сглаживание
    if (isnan(rssiEMA)) rssiEMA = rssi;
    else rssiEMA = ALPHA * rssi + (1.0f - ALPHA) * rssiEMA;

    // Расстояние по модели
    float dist_m = powf(10.0f, (float)(TXPWR_1M_DB - rssiEMA) / (10.0f * N_PLF));

    // Вывод расстояния на serial
    Serial.printf("Target %s found! RSSI: %d, Smoothed RSSI: %.2f, Distance: %.3f meters\n", 
                  name.c_str(), rssi, rssiEMA, dist_m);
  }
};

void setup() {
  Serial.begin(115200);
  delay(200);

  BLEDevice::init("ANCHOR");  // Инициализация устройства как якоря
  BLEDevice::setPower(ESP_PWR_LVL_P9); // стабильнее ловим рекламу

  BLEScan *scan = BLEDevice::getScan();
  scan->setAdvertisedDeviceCallbacks(new AdvCallbacks(), true); // обработка пакетов
  scan->setActiveScan(true);  // активное сканирование

  uint16_t intv = (uint16_t)(SCAN_INTERVAL_MS / 0.625);
  uint16_t wnd  = (uint16_t)(SCAN_WINDOW_MS   / 0.625);
  scan->setInterval(intv);
  scan->setWindow(wnd);

  // Начинаем непрерывное сканирование
  scan->start(SCAN_DURATION_S, false);

  Serial.println("Anchor device is ready and scanning for target BLE devices...");
}

void loop() {
  // библиотека сама крутит onResult в фоне при активном скане
  delay(10);
}
