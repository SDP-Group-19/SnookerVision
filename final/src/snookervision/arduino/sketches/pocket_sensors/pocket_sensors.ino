namespace {

constexpr uint8_t kPocketCount = 6;
constexpr unsigned long kDebounceDelayMs = 40;

struct PocketSensor {
  uint8_t id;
  uint8_t pin;
  bool lastReading;
  bool stableState;
  unsigned long lastDebounceAt;
};

PocketSensor pockets[] = {
    {1, 2, HIGH, HIGH, 0},
    {2, 3, HIGH, HIGH, 0},
    {3, 4, HIGH, HIGH, 0},
    {4, 5, HIGH, HIGH, 0},
    {5, 6, HIGH, HIGH, 0},
    {6, 7, HIGH, HIGH, 0},
};

void printPocketStates() {
  for (uint8_t i = 0; i < kPocketCount; ++i) {
    Serial.print(F("POCKET_STATE "));
    Serial.print(pockets[i].id);
    Serial.print(' ');
    Serial.println(pockets[i].stableState == LOW ? F("ACTIVE") : F("IDLE"));
  }
}

void initializeSensors() {
  for (uint8_t i = 0; i < kPocketCount; ++i) {
    pinMode(pockets[i].pin, INPUT_PULLUP);

    const bool initialState = digitalRead(pockets[i].pin);
    pockets[i].lastReading = initialState;
    pockets[i].stableState = initialState;
    pockets[i].lastDebounceAt = 0;
  }
}

void handlePocket(PocketSensor &pocket) {
  const bool reading = digitalRead(pocket.pin);

  if (reading != pocket.lastReading) {
    pocket.lastDebounceAt = millis();
  }

  if ((millis() - pocket.lastDebounceAt) > kDebounceDelayMs && reading != pocket.stableState) {
    pocket.stableState = reading;

    if (pocket.stableState == LOW) {
      Serial.print(F("POCKET "));
      Serial.println(pocket.id);
    } else {
      Serial.print(F("POCKET_CLEAR "));
      Serial.println(pocket.id);
    }
  }

  pocket.lastReading = reading;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  initializeSensors();

  Serial.println(F("READY POCKETS"));
  printPocketStates();
}

void loop() {
  for (uint8_t i = 0; i < kPocketCount; ++i) {
    handlePocket(pockets[i]);
  }
}
