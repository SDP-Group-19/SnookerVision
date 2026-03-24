#include <Arduino.h>
#include <ChainableLED.h>
#include <TM1637Display.h>
#include <Wire.h>
#include <rgb_lcd.h>

/*
Serial commands:
SET 1 12
SET 2 34
LCD Player 1|At table
LIGHT 1 255 0 0
LIGHT 2 0 255 0
LIGHTOFF
RESET
STATUS
*/

namespace {

// Hardware map: change these pin numbers to match your wiring.
constexpr uint8_t kDisplayCount = 2;
constexpr uint8_t kMaxScore = 9999;
constexpr unsigned long kDebounceDelayMs = 40;
constexpr uint8_t kBrightness = 0x0f;
constexpr uint8_t kRgbLedCount = 2;
constexpr uint8_t kLcdColumns = 16;
constexpr uint8_t kLcdRows = 2;

constexpr uint8_t kDisplay1ClkPin = 2;
constexpr uint8_t kDisplay1DioPin = 3;
constexpr uint8_t kDisplay2ClkPin = 4;
constexpr uint8_t kDisplay2DioPin = 5;

constexpr uint8_t kDisplay1UpButtonPin = 6;
constexpr uint8_t kDisplay1DownButtonPin = 13;
constexpr uint8_t kDisplay2UpButtonPin = 8;
constexpr uint8_t kDisplay2DownButtonPin = 22;
constexpr uint8_t kFullResetButtonPin = 9;
constexpr uint8_t kLastPositionButtonPin = 10;
constexpr uint8_t kChangePlayerButtonPin = 7;

constexpr uint8_t kTurnLightDataPin = 11;
constexpr uint8_t kTurnLightClockPin = 12;
constexpr uint8_t kTurnLight1Index = 0;
constexpr uint8_t kTurnLight2Index = 1;

enum class ButtonAction : uint8_t {
  AdjustScore,
  ResetScores,
  TogglePlayer,
  PrintOnly,
};

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
  ButtonAction action;
  uint8_t displayIndex;
  int delta;
  const char *message;
  bool lastReading;
  bool stableState;
  unsigned long lastDebounceAt;
};

ScoreDisplay displays[] = {
    ScoreDisplay(1, kDisplay1ClkPin, kDisplay1DioPin),
    ScoreDisplay(2, kDisplay2ClkPin, kDisplay2DioPin),
};

ButtonState buttons[] = {
    {kDisplay1UpButtonPin, ButtonAction::AdjustScore, 0, +1, "BUTTON DISPLAY_1_UP", HIGH, HIGH, 0},
    {kDisplay1DownButtonPin, ButtonAction::AdjustScore, 0, -1, "BUTTON DISPLAY_1_DOWN", HIGH, HIGH, 0},
    {kDisplay2UpButtonPin, ButtonAction::AdjustScore, 1, +1, "BUTTON DISPLAY_2_UP", HIGH, HIGH, 0},
    {kDisplay2DownButtonPin, ButtonAction::AdjustScore, 1, -1, "BUTTON DISPLAY_2_DOWN", HIGH, HIGH, 0},
    {kFullResetButtonPin, ButtonAction::ResetScores, 0, 0, "BUTTON FULL_RESET", HIGH, HIGH, 0},
    {kLastPositionButtonPin, ButtonAction::PrintOnly, 0, 0, "BUTTON LAST_POSITION", HIGH, HIGH, 0},
    {kChangePlayerButtonPin, ButtonAction::TogglePlayer, 0, 0, "BUTTON CHANGE_PLAYER", HIGH, HIGH, 0},
};

constexpr uint8_t kButtonCount = sizeof(buttons) / sizeof(buttons[0]);

String inputBuffer;
rgb_lcd lcd;
ChainableLED rgbLeds(kTurnLightDataPin, kTurnLightClockPin, kRgbLedCount);
uint8_t activeTurnLightIndex = kTurnLight2Index;

bool parseLong(const String &value, long &result);

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
  Serial.println(F("  LCD <line1>|<line2>"));
  Serial.println(F("  LCDCLEAR"));
  Serial.println(F("  LIGHT <1|2|ALL> <r> <g> <b>"));
  Serial.println(F("  LIGHTOFF"));
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

String fitLcdLine(String text) {
  if (text.length() > kLcdColumns) {
    return text.substring(0, kLcdColumns);
  }

  while (text.length() < kLcdColumns) {
    text += ' ';
  }

  return text;
}

void setLcdMessage(const String &line1, const String &line2) {
  lcd.setCursor(0, 0);
  lcd.print(fitLcdLine(line1));
  lcd.setCursor(0, 1);
  lcd.print(fitLcdLine(line2));

  Serial.print(F("LCD "));
  Serial.print(line1);
  Serial.print(F(" | "));
  Serial.println(line2);
}

void clearLcd() {
  setLcdMessage("", "");
}

bool isValidLedIndex(long ledId) {
  return ledId >= 1 && ledId <= kRgbLedCount;
}

uint8_t clampColor(long value) {
  if (value < 0) {
    return 0;
  }

  if (value > 255) {
    return 255;
  }

  return static_cast<uint8_t>(value);
}

void setRgbLed(uint8_t ledIndex, uint8_t red, uint8_t green, uint8_t blue) {
  rgbLeds.setColorRGB(ledIndex, red, green, blue);
}

const __FlashStringHelper *turnLightName(uint8_t ledIndex) {
  if (ledIndex == kTurnLight1Index) {
    return F("TURN_LIGHT_1");
  }

  return F("TURN_LIGHT_2");
}

void setAllRgbLeds(uint8_t red, uint8_t green, uint8_t blue) {
  for (uint8_t i = 0; i < kRgbLedCount; ++i) {
    setRgbLed(i, red, green, blue);
  }
}

void showActiveTurnLight() {
  setAllRgbLeds(0, 0, 0);
  setRgbLed(activeTurnLightIndex, 255, 255, 255);

  Serial.print(F("ACTIVE_PLAYER "));
  Serial.println(turnLightName(activeTurnLightIndex));
}

void toggleActiveTurnLight() {
  if (activeTurnLightIndex == kTurnLight1Index) {
    activeTurnLightIndex = kTurnLight2Index;
  } else {
    activeTurnLightIndex = kTurnLight1Index;
  }

  showActiveTurnLight();
}

bool parseLedCommand(const String &payload) {
  const int firstSpace = payload.indexOf(' ');
  if (firstSpace < 0) {
    return false;
  }

  const String target = payload.substring(0, firstSpace);
  String remainder = payload.substring(firstSpace + 1);
  remainder.trim();

  const int secondSpace = remainder.indexOf(' ');
  if (secondSpace < 0) {
    return false;
  }

  const String redText = remainder.substring(0, secondSpace);
  remainder = remainder.substring(secondSpace + 1);
  remainder.trim();

  const int thirdSpace = remainder.indexOf(' ');
  if (thirdSpace < 0) {
    return false;
  }

  const String greenText = remainder.substring(0, thirdSpace);
  const String blueText = remainder.substring(thirdSpace + 1);

  long redValue = 0;
  long greenValue = 0;
  long blueValue = 0;
  if (!parseLong(redText, redValue) || !parseLong(greenText, greenValue) || !parseLong(blueText, blueValue)) {
    return false;
  }

  const uint8_t red = clampColor(redValue);
  const uint8_t green = clampColor(greenValue);
  const uint8_t blue = clampColor(blueValue);

  if (target == "ALL") {
    setAllRgbLeds(red, green, blue);
    Serial.print(F("TURN_LIGHT ALL "));
    Serial.print(red);
    Serial.print(' ');
    Serial.print(green);
    Serial.print(' ');
    Serial.println(blue);
    return true;
  }

  long ledId = 0;
  if (!parseLong(target, ledId) || !isValidLedIndex(ledId)) {
    return false;
  }

  const uint8_t ledIndex = static_cast<uint8_t>(ledId - 1);
  setRgbLed(ledIndex, red, green, blue);
  Serial.print(turnLightName(ledIndex));
  Serial.print(' ');
  Serial.print(red);
  Serial.print(' ');
  Serial.print(green);
  Serial.print(' ');
  Serial.println(blue);
  return true;
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

  if (command.length() == 0) {
    return;
  }

  if (command.indexOf('=') >= 0) {
    processLegacyCommand(command);
    return;
  }

  const int firstSpace = command.indexOf(' ');
  String action = firstSpace < 0 ? command : command.substring(0, firstSpace);
  String payload = firstSpace < 0 ? "" : command.substring(firstSpace + 1);
  action.toUpperCase();
  payload.trim();

  if (action == "RESET") {
    resetScores();
    return;
  }

  if (action == "STATUS") {
    reportAllScores();
    return;
  }

  if (action == "LCD") {
    const int separator = payload.indexOf('|');
    if (separator < 0) {
      setLcdMessage(payload, "");
      return;
    }

    const String line1 = payload.substring(0, separator);
    const String line2 = payload.substring(separator + 1);
    setLcdMessage(line1, line2);
    return;
  }

  if (action == "LCDCLEAR") {
    clearLcd();
    return;
  }

  if (action == "LIGHTOFF" || action == "LEDOFF") {
    setAllRgbLeds(0, 0, 0);
    Serial.println(F("TURN_LIGHT ALL 0 0 0"));
    return;
  }

  if (action == "LIGHT" || action == "LED") {
    String normalizedPayload = payload;
    normalizedPayload.trim();
    const int targetSpace = normalizedPayload.indexOf(' ');
    if (targetSpace > 0) {
      String target = normalizedPayload.substring(0, targetSpace);
      target.toUpperCase();
      normalizedPayload = target + normalizedPayload.substring(targetSpace);
    } else {
      normalizedPayload.toUpperCase();
    }

    if (!parseLedCommand(normalizedPayload)) {
      Serial.println(F("ERR Use LIGHT <1|2|ALL> <r> <g> <b>"));
    }
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
      if (button.action == ButtonAction::AdjustScore) {
        applyScoreChange(button.displayIndex, button.delta, F("ADD"));
      } else if (button.action == ButtonAction::ResetScores) {
        resetScores();
      } else if (button.action == ButtonAction::TogglePlayer) {
        toggleActiveTurnLight();
      }

      Serial.println(button.message);
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
}

void initializeDisplays() {
  for (uint8_t i = 0; i < kDisplayCount; ++i) {
    displays[i].driver.setBrightness(kBrightness);
    showDisplay(displays[i]);
  }
}

void initializeLcd() {
  lcd.begin(kLcdColumns, kLcdRows);
  lcd.setRGB(255, 255, 255);
  setLcdMessage("SnookerVision", "Controller ready");
}

void initializeRgbLeds() {
  showActiveTurnLight();
}

}  // namespace

void setup() {
  Serial.begin(115200);
  initializeButtons();
  initializeDisplays();
  initializeLcd();
  initializeRgbLeds();

  Serial.println(F("READY DISPLAY"));
  printHelp();
  reportAllScores();
}

void loop() {
  handleSerialInput();

  for (uint8_t i = 0; i < kButtonCount; ++i) {
    handleButton(buttons[i]);
  }
}
