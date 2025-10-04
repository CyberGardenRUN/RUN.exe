#include <Arduino.h>
#include <NimBLEDevice.h>

static const char* TAG_NAME = "TAG_123";

// Интервал рекламы 100 мс (единица 0.625 мс)
static const uint16_t ADV_INT = 0x00A0;

// Короткие manufacturer data (чем короче — тем лучше для коллизий/джиттера)
static const uint8_t mfgData[] = { 0x12, 0x34, 'T', '1' };

void setup() {
  Serial.begin(115200);
  delay(50);

  // Инициализация BLE (имя добавим в adv data)
  NimBLEDevice::init("");
  NimBLEDevice::setPower(ESP_PWR_LVL_P9);        // макс. мощность ADV

  // По возможности используем постоянный публичный адрес (стабильнее для фильтрации)
  // Если у вас другой кейс — эту строку можно закомментировать.
  NimBLEDevice::setOwnAddrType(BLE_OWN_ADDR_PUBLIC);

  // Формируем advertising payload
  NimBLEAdvertisementData adv;
  adv.setFlags(0x06);                            // LE General Discoverable + BR/EDR Not Supported
  adv.setName(TAG_NAME);                         // имя прямо в advertising
  adv.setManufacturerData(std::string((const char*)mfgData, sizeof(mfgData)));

  NimBLEAdvertising* pAdv = NimBLEDevice::getAdvertising();
  pAdv->setAdvertisementData(adv);
  // Scan Response НЕ задаём -> получается non-scannable реклама
  pAdv->setMinInterval(ADV_INT);
  pAdv->setMaxInterval(ADV_INT);
  // Каналы по умолчанию 37/38/39 — оставляем как есть (самый устойчивый вариант)

  pAdv->start();
  Serial.println("NimBLE TAG advertising started (100 ms, non-scannable, max TX).");
}

void loop() {
  // ничего не делаем — радио пусть работает ровно
}
