#if defined(ESP8266)
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ESP8266mDNS.h>
using WebServerType = ESP8266WebServer;
#else
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
using WebServerType = WebServer;
#endif

#include <Wire.h>
#include <TM1637Display.h>
#include <rgb_lcd.h>

/*
Scoreboard node.

Hardware on this ESP8266:
- left 4-digit display on D2/D3
- right 4-digit display on D4/D5
- left score up/down buttons on D6/D7
- last position button on D8
- 16x2 LCD on configurable I2C pins

Network commands:
- SET 1 12
- SET 2 34
- ADD 1 1
- ADD 2 -1
- RESET
- STATUS
- LCD Player 1|At table
- LCDCLEAR
*/

namespace {

const char *kWifiSsid = "YOUR_WIFI_NAME";
const char *kWifiPassword = "YOUR_WIFI_PASSWORD";
const char *kHostname = "snooker-display-1";

constexpr uint8_t kDisplay1ClkPin = 13;    // D2
constexpr uint8_t kDisplay1DioPin = 10;    // D3
constexpr uint8_t kDisplay2ClkPin = 15;    // D4
constexpr uint8_t kDisplay2DioPin = 2;     // D5
constexpr uint8_t kLeftUpButtonPin = 5;    // D6
constexpr uint8_t kLeftDownButtonPin = 4;  // D7
constexpr uint8_t kLastPositionButtonPin = 0;  // D8, boot-strapping pin
const uint8_t kLcdSdaPin = SDA;
const uint8_t kLcdSclPin = SCL;
constexpr uint8_t kBrightness = 0x0f;
constexpr int kMaxScore = 9999;
constexpr uint8_t kLcdColumns = 16;
constexpr unsigned long kDebounceDelayMs = 40;
constexpr unsigned long kReconnectIntervalMs = 5000;

enum class ButtonAction : uint8_t {
  AddLeft,
  LastPosition,
};

struct ButtonState {
  uint8_t pin;
  ButtonAction action;
  int delta;
  const char *message;
  bool activeLow;
  bool lastReading;
  bool stableState;
  unsigned long lastDebounceAt;
};

TM1637Display display1(kDisplay1ClkPin, kDisplay1DioPin);
TM1637Display display2(kDisplay2ClkPin, kDisplay2DioPin);
rgb_lcd lcd;
WebServerType server(80);

ButtonState buttons[] = {
    {kLeftUpButtonPin, ButtonAction::AddLeft, +1, "BUTTON DISPLAY_1_UP", true, HIGH, HIGH, 0},
    {kLeftDownButtonPin, ButtonAction::AddLeft, -1, "BUTTON DISPLAY_1_DOWN", true, HIGH, HIGH, 0},
    {kLastPositionButtonPin, ButtonAction::LastPosition, 0, "BUTTON LAST_POSITION", false, LOW, LOW, 0},
};

constexpr uint8_t kButtonCount = sizeof(buttons) / sizeof(buttons[0]);

int score1 = 0;
int score2 = 0;
String inputBuffer;
String currentLcdLine1 = "";
String currentLcdLine2 = "";
String pendingEvents = "";
unsigned long lastWifiAttemptAt = 0;

int clampScore(int score) {
  if (score < 0) {
    return 0;
  }

  if (score > kMaxScore) {
    return kMaxScore;
  }

  return score;
}

bool parseLong(const String &value, long &result) {
  if (value.length() == 0 || value.length() > 15) {
    return false;
  }

  char buffer[16];
  value.toCharArray(buffer, sizeof(buffer));

  char *endPtr = nullptr;
  result = strtol(buffer, &endPtr, 10);
  return endPtr != buffer && *endPtr == '\0';
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

void refreshDisplays() {
  display1.showNumberDec(score1, false);
  display2.showNumberDec(score2, false);
}

void applyLcdMessage(const String &line1, const String &line2) {
  currentLcdLine1 = line1;
  currentLcdLine2 = line2;

  lcd.setCursor(0, 0);
  lcd.print(fitLcdLine(line1));
  lcd.setCursor(0, 1);
  lcd.print(fitLcdLine(line2));
}

String buildStatus() {
  String status = "NODE SCOREBOARD HOSTNAME ";
  status += kHostname;
  status += " IP ";
  status += WiFi.localIP().toString();
  status += " SCORE1 ";
  status += score1;
  status += " SCORE2 ";
  status += score2;
  status += " LCD ";
  status += currentLcdLine1;
  status += "|";
  status += currentLcdLine2;
  return status;
}

void setScore(uint8_t displayId, int value) {
  if (displayId == 1) {
    score1 = clampScore(value);
  } else if (displayId == 2) {
    score2 = clampScore(value);
  }

  refreshDisplays();
}

void addScore(uint8_t displayId, int delta) {
  if (displayId == 1) {
    score1 = clampScore(score1 + delta);
  } else if (displayId == 2) {
    score2 = clampScore(score2 + delta);
  }

  refreshDisplays();
}

String handleScoreCommand(const String &payload, bool additive) {
  const int separator = payload.indexOf(' ');
  if (separator < 0) {
    return "ERR Missing arguments";
  }

  String left = payload.substring(0, separator);
  String right = payload.substring(separator + 1);
  left.trim();
  right.trim();

  long displayId = 0;
  long value = 0;
  if (!parseLong(left, displayId) || !parseLong(right, value)) {
    return "ERR Invalid arguments";
  }

  if (displayId != 1 && displayId != 2) {
    return "ERR Invalid display";
  }

  if (additive) {
    addScore(static_cast<uint8_t>(displayId), static_cast<int>(value));
  } else {
    setScore(static_cast<uint8_t>(displayId), static_cast<int>(value));
  }

  return buildStatus();
}

String executeCommand(String command) {
  command.trim();
  if (command.length() == 0) {
    return "ERR Empty command";
  }

  if (command.indexOf('=') >= 0) {
    const int equalsAt = command.indexOf('=');
    String left = command.substring(0, equalsAt);
    String right = command.substring(equalsAt + 1);
    left.trim();
    right.trim();
    return handleScoreCommand(left + " " + right, false);
  }

  const int firstSpace = command.indexOf(' ');
  String action = firstSpace < 0 ? command : command.substring(0, firstSpace);
  String payload = firstSpace < 0 ? "" : command.substring(firstSpace + 1);
  action.toUpperCase();
  payload.trim();

  if (action == "SET") {
    return handleScoreCommand(payload, false);
  }

  if (action == "ADD") {
    return handleScoreCommand(payload, true);
  }

  if (action == "RESET") {
    score1 = 0;
    score2 = 0;
    refreshDisplays();
    return buildStatus();
  }

  if (action == "STATUS") {
    return buildStatus();
  }

  if (action == "LCD") {
    const int separator = payload.indexOf('|');
    if (separator < 0) {
      applyLcdMessage(payload, "");
    } else {
      applyLcdMessage(payload.substring(0, separator), payload.substring(separator + 1));
    }
    return buildStatus();
  }

  if (action == "LCDCLEAR") {
    applyLcdMessage("", "");
    return buildStatus();
  }

  if (action == "LIGHT" || action == "LIGHTOFF" || action == "LEDOFF") {
    return "OK Ignored light command";
  }

  return "ERR Unknown command";
}

void handleCommandRequest() {
  const String response = executeCommand(server.arg("plain"));
  server.send(200, "text/plain", response);
  Serial.println(response);
}

void handleStatusRequest() {
  server.send(200, "text/plain", buildStatus());
}

void enqueueEvent(const String &eventName) {
  if (pendingEvents.length() > 0) {
    pendingEvents += '\n';
  }
  pendingEvents += eventName;
}

void handleEventsRequest() {
  const String response = pendingEvents;
  pendingEvents = "";
  server.send(200, "text/plain", response);
}

void ensureWifiConnected() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  const unsigned long now = millis();
  if (now - lastWifiAttemptAt < kReconnectIntervalMs) {
    return;
  }

  lastWifiAttemptAt = now;
  WiFi.disconnect();
  WiFi.begin(kWifiSsid, kWifiPassword);
}

void connectWifiBlocking() {
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(kWifiSsid, kWifiPassword);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
}

void setupServer() {
  server.on("/status", HTTP_GET, handleStatusRequest);
  server.on("/events", HTTP_GET, handleEventsRequest);
  server.on("/command", HTTP_POST, handleCommandRequest);
  server.on("/command", HTTP_GET, []() {
    const String response = executeCommand(server.arg("cmd"));
    server.send(200, "text/plain", response);
    Serial.println(response);
  });
  server.begin();
}

void initializeButtons() {
  for (uint8_t i = 0; i < kButtonCount; ++i) {
    if (buttons[i].activeLow) {
      pinMode(buttons[i].pin, INPUT_PULLUP);
    } else {
      pinMode(buttons[i].pin, INPUT);
    }

    const bool initial = digitalRead(buttons[i].pin);
    buttons[i].lastReading = initial;
    buttons[i].stableState = initial;
    buttons[i].lastDebounceAt = 0;
  }
}

bool isPressed(const ButtonState &button) {
  return button.activeLow ? button.stableState == LOW : button.stableState == HIGH;
}

void handleButton(ButtonState &button) {
  const bool reading = digitalRead(button.pin);
  if (reading != button.lastReading) {
    button.lastDebounceAt = millis();
  }

  if ((millis() - button.lastDebounceAt) > kDebounceDelayMs && reading != button.stableState) {
    button.stableState = reading;

    if (isPressed(button)) {
      if (button.action == ButtonAction::AddLeft) {
        addScore(1, button.delta);
      } else if (button.action == ButtonAction::LastPosition) {
        enqueueEvent("LAST_POSITION");
      }

      Serial.println(button.message);
    }
  }

  button.lastReading = reading;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  display1.setBrightness(kBrightness);
  display2.setBrightness(kBrightness);
  refreshDisplays();

  Wire.begin(kLcdSdaPin, kLcdSclPin);
  lcd.begin(16, 2);
  applyLcdMessage("SnookerVision", "Ready");
  initializeButtons();

  connectWifiBlocking();
  MDNS.begin(kHostname);
  setupServer();

  Serial.println(buildStatus());
}

void loop() {
  ensureWifiConnected();
  server.handleClient();
  MDNS.update();

  for (uint8_t i = 0; i < kButtonCount; ++i) {
    handleButton(buttons[i]);
  }
}
