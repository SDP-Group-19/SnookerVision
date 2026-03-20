#include <Arduino.h>
#include <TM1637Display.h>

namespace {

constexpr uint8_t kDisplayCount = 2;
constexpr uint8_t kButtonCount = 4;
constexpr uint8_t kMaxScore = 9999;
constexpr unsigned long kDebounceDelayMs = 40;
constexpr uint8_t kBrightness = 0x0f;
constexpr uint8_t kResetButtonPin = 11;

struct ScoreDisplay {
  uint8_t id;
  uint8_t clkPin;
  uint8_t dioPin;
  int score;
  TM1637Display driver;

  ScoreDisplay(uint8_t displayId, uint8_t clk, uint8_t dio)
      : id(displayId), clkPin(clk), dioPin(dio), score(0), driver(clk, dio) {}
};

struct ButtonState {
  uint8_t pin;
  uint8_t displayIndex;
  int delta;
  const char *label;
  bool lastReading;
  bool stableState;
  unsigned long lastDebounceAt;
};

ScoreDisplay displays[] = {
    ScoreDisplay(1, 2, 3),
    ScoreDisplay(2, 4, 5),
};

ButtonState buttons[] = {
    {6, 0, +1, "D1_UP", HIGH, HIGH, 0},
    {7, 0, -1, "D1_DOWN", HIGH, HIGH, 0},
    {8, 1, +1, "D2_UP", HIGH, HIGH, 0},
    {10, 1, -1, "D2_DOWN", HIGH, HIGH, 0},
};

String inputBuffer;

int clampScore(int score) {
  if (score < 0) {
    return 0;
  }

  if (score > kMaxScore) {
    return kMaxScore;
  }

  return score;
}

void printHelp() {
  Serial.println(F("Commands:"));
  Serial.println(F("  SET <display> <score>"));
  Serial.println(F("  ADD <display> <delta>"));
  Serial.println(F("  RESET"));
  Serial.println(F("  STATUS"));
  Serial.println(F("Legacy format still supported: 1=1234 or 2=5678"));
}

void showDisplay(ScoreDisplay &display) {
  display.driver.showNumberDec(display.score, false);
}

void printScoreState(const ScoreDisplay &display) {
  Serial.print(F("SCORE "));
  Serial.print(display.id);
  Serial.print(' ');
  Serial.println(display.score);
}

void refreshAllDisplays() {
  for (uint8_t i = 0; i < kDisplayCount; ++i) {
    showDisplay(displays[i]);
  }
}

void reportAllScores() {
  for (uint8_t i = 0; i < kDisplayCount; ++i) {
    printScoreState(displays[i]);
  }
}

bool isValidDisplayId(long displayId) {
  return displayId >= 1 && displayId <= kDisplayCount;
}

void applyScoreChange(uint8_t displayIndex, int delta, const __FlashStringHelper *reason) {
  ScoreDisplay &display = displays[displayIndex];
  display.score = clampScore(display.score + delta);
  showDisplay(display);

  Serial.print(reason);
  Serial.print(' ');
  Serial.print(display.id);
  Serial.print(' ');
  Serial.println(display.score);
}

void setScore(uint8_t displayIndex, int score, const __FlashStringHelper *reason) {
  ScoreDisplay &display = displays[displayIndex];
  display.score = clampScore(score);
  showDisplay(display);

  Serial.print(reason);
  Serial.print(' ');
  Serial.print(display.id);
  Serial.print(' ');
  Serial.println(display.score);
}

void resetScores() {
  for (uint8_t i = 0; i < kDisplayCount; ++i) {
    displays[i].score = 0;
  }

  refreshAllDisplays();
  Serial.println(F("RESET OK"));
  reportAllScores();
}

bool parseLong(const String &value, long &result) {
  if (value.length() == 0) {
    return false;
  }

  char buffer[16];
  const size_t maxLength = sizeof(buffer) - 1;
  if (value.length() > maxLength) {
    return false;
  }

  const size_t copyLength = value.length() < maxLength ? value.length() : maxLength;
  value.substring(0, copyLength).toCharArray(buffer, copyLength + 1);

  char *endPtr = nullptr;
  result = strtol(buffer, &endPtr, 10);
  return endPtr != buffer && *endPtr == '\0';
}

void processLegacyCommand(String command) {
  const int equalsAt = command.indexOf('=');
  if (equalsAt < 0) {
    Serial.println(F("ERR Unknown command"));
    return;
  }

  const String left = command.substring(0, equalsAt);
  const String right = command.substring(equalsAt + 1);

  long displayId = 0;
  long score = 0;
  if (!parseLong(left, displayId) || !parseLong(right, score) || !isValidDisplayId(displayId)) {
    Serial.println(F("ERR Use 1=1234 or 2=5678"));
    return;
  }

  setScore(static_cast<uint8_t>(displayId - 1), static_cast<int>(score), F("SET"));
}

void processCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command.length() == 0) {
    return;
  }

  if (command.indexOf('=') >= 0) {
    processLegacyCommand(command);
    return;
  }

  const int firstSpace = command.indexOf(' ');
  const String action = firstSpace < 0 ? command : command.substring(0, firstSpace);
  String payload = firstSpace < 0 ? "" : command.substring(firstSpace + 1);
  payload.trim();

  if (action == "RESET") {
    resetScores();
    return;
  }

  if (action == "STATUS") {
    reportAllScores();
    return;
  }

  const int secondSpace = payload.indexOf(' ');
  if (secondSpace < 0) {
    Serial.println(F("ERR Missing arguments"));
    return;
  }

  String left = payload.substring(0, secondSpace);
  String right = payload.substring(secondSpace + 1);
  left.trim();
  right.trim();

  long displayId = 0;
  long value = 0;
  if (!parseLong(left, displayId) || !parseLong(right, value) || !isValidDisplayId(displayId)) {
    Serial.println(F("ERR Invalid arguments"));
    return;
  }

  const uint8_t displayIndex = static_cast<uint8_t>(displayId - 1);
  if (action == "SET") {
    setScore(displayIndex, static_cast<int>(value), F("SET"));
    return;
  }

  if (action == "ADD") {
    applyScoreChange(displayIndex, static_cast<int>(value), F("ADD"));
    return;
  }

  Serial.println(F("ERR Unknown command"));
}

void handleSerialInput() {
  while (Serial.available() > 0) {
    const char incoming = static_cast<char>(Serial.read());

    if (incoming == '\r' || incoming == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
      continue;
    }

    if (isPrintable(incoming) && inputBuffer.length() < 48) {
      inputBuffer += incoming;
    }
  }
}

void handleButton(ButtonState &button) {
  const bool reading = digitalRead(button.pin);

  if (reading != button.lastReading) {
    button.lastDebounceAt = millis();
  }

  if ((millis() - button.lastDebounceAt) > kDebounceDelayMs && reading != button.stableState) {
    button.stableState = reading;

    if (button.stableState == LOW) {
      applyScoreChange(button.displayIndex, button.delta, F("ADD"));

      Serial.print(F("BUTTON "));
      Serial.println(button.label);
    }
  }

  button.lastReading = reading;
}

void initializeButtons() {
  for (uint8_t i = 0; i < kButtonCount; ++i) {
    pinMode(buttons[i].pin, INPUT_PULLUP);
    const bool initialState = digitalRead(buttons[i].pin);
    buttons[i].lastReading = initialState;
    buttons[i].stableState = initialState;
    buttons[i].lastDebounceAt = 0;
  }

  pinMode(kResetButtonPin, INPUT_PULLUP);
}

void initializeDisplays() {
  for (uint8_t i = 0; i < kDisplayCount; ++i) {
    displays[i].driver.setBrightness(kBrightness);
    showDisplay(displays[i]);
  }
}

void handleResetButton() {
  static bool lastReading = HIGH;
  static bool stableState = HIGH;
  static unsigned long lastDebounceAt = 0;

  const bool reading = digitalRead(kResetButtonPin);
  if (reading != lastReading) {
    lastDebounceAt = millis();
  }

  if ((millis() - lastDebounceAt) > kDebounceDelayMs && reading != stableState) {
    stableState = reading;

    if (stableState == LOW) {
      resetScores();
      Serial.println(F("BUTTON RESET"));
    }
  }

  lastReading = reading;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  initializeButtons();
  initializeDisplays();

  Serial.println(F("READY DISPLAY"));
  printHelp();
  reportAllScores();
}

void loop() {
  handleSerialInput();

  for (uint8_t i = 0; i < kButtonCount; ++i) {
    handleButton(buttons[i]);
  }

  handleResetButton();
}
