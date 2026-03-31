#if defined(ESP8266)
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ESP8266mDNS.h>
#include <ESP8266HTTPClient.h>
using WebServerType = ESP8266WebServer;
#else
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <HTTPClient.h>
using WebServerType = WebServer;
#endif

#include <ChainableLED.h>

/*
Control node.

Hardware on this ESP8266:
- display right up/down buttons on D2/D3
- chainable LED turn lights on D4/D5
- change player button on D6
- full reset button on D7

This node forwards score/reset button actions to the scoreboard node over Wi-Fi.
*/

namespace {

const char *kWifiSsid = "YOUR_WIFI_NAME";
const char *kWifiPassword = "YOUR_WIFI_PASSWORD";
const char *kHostname = "snooker-display-2";
const char *kScoreboardBaseUrl = "http://snooker-display-1.local";

constexpr uint8_t kRightUpButtonPin = 13;       // D2
constexpr uint8_t kRightDownButtonPin = 10;     // D3
constexpr uint8_t kLightDataPin = 15;           // D4
constexpr uint8_t kLightClockPin = 2;           // D5
constexpr uint8_t kChangePlayerButtonPin = 5;   // D6
constexpr uint8_t kFullResetButtonPin = 4;      // D7
constexpr uint8_t kRgbLedCount = 2;
constexpr uint8_t kPlayer1LightIndex = 0;
constexpr uint8_t kPlayer2LightIndex = 1;
constexpr unsigned long kDebounceDelayMs = 40;
constexpr unsigned long kReconnectIntervalMs = 5000;

enum class ButtonAction : uint8_t {
  AddRight,
  TogglePlayer,
  ResetAll,
};

struct ButtonState {
  uint8_t pin;
  ButtonAction action;
  int delta;
  const char *message;
  bool lastReading;
  bool stableState;
  unsigned long lastDebounceAt;
};

WebServerType server(80);
ChainableLED turnLights(kLightDataPin, kLightClockPin, kRgbLedCount);

ButtonState buttons[] = {
    {kRightUpButtonPin, ButtonAction::AddRight, +1, "BUTTON DISPLAY_2_UP", HIGH, HIGH, 0},
    {kRightDownButtonPin, ButtonAction::AddRight, -1, "BUTTON DISPLAY_2_DOWN", HIGH, HIGH, 0},
    {kChangePlayerButtonPin, ButtonAction::TogglePlayer, 0, "BUTTON CHANGE_PLAYER", HIGH, HIGH, 0},
    {kFullResetButtonPin, ButtonAction::ResetAll, 0, "BUTTON FULL_RESET", HIGH, HIGH, 0},
};

constexpr uint8_t kButtonCount = sizeof(buttons) / sizeof(buttons[0]);

uint8_t currentLight1Red = 0;
uint8_t currentLight1Green = 0;
uint8_t currentLight1Blue = 0;
uint8_t currentLight2Red = 0;
uint8_t currentLight2Green = 0;
uint8_t currentLight2Blue = 0;
uint8_t activePlayer = 1;
unsigned long lastWifiAttemptAt = 0;

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

uint8_t clampColor(long value) {
  if (value < 0) {
    return 0;
  }

  if (value > 255) {
    return 255;
  }

  return static_cast<uint8_t>(value);
}

void setLight(uint8_t lightIndex, uint8_t red, uint8_t green, uint8_t blue) {
  turnLights.setColorRGB(lightIndex, red, green, blue);

  if (lightIndex == kPlayer1LightIndex) {
    currentLight1Red = red;
    currentLight1Green = green;
    currentLight1Blue = blue;
  } else {
    currentLight2Red = red;
    currentLight2Green = green;
    currentLight2Blue = blue;
  }
}

void setAllLights(uint8_t red, uint8_t green, uint8_t blue) {
  setLight(kPlayer1LightIndex, red, green, blue);
  setLight(kPlayer2LightIndex, red, green, blue);
}

void showActivePlayerLight() {
  setAllLights(0, 0, 0);

  if (activePlayer == 1) {
    setLight(kPlayer1LightIndex, 40, 40, 40);
  } else {
    setLight(kPlayer2LightIndex, 40, 40, 40);
  }
}

bool postToScoreboard(const String &command) {
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  WiFiClient client;
  HTTPClient http;
  const String endpoint = String(kScoreboardBaseUrl) + "/command";

#if defined(ESP8266)
  if (!http.begin(client, endpoint)) {
    return false;
  }
#else
  if (!http.begin(endpoint)) {
    return false;
  }
#endif

  http.addHeader("Content-Type", "text/plain; charset=utf-8");
  const int statusCode = http.POST(command);
  http.end();
  return statusCode > 0 && statusCode < 400;
}

String buildStatus() {
  String status = "NODE CONTROL HOSTNAME ";
  status += kHostname;
  status += " IP ";
  status += WiFi.localIP().toString();
  status += " ACTIVE_PLAYER ";
  status += activePlayer;
  status += " LIGHT1 ";
  status += currentLight1Red;
  status += ",";
  status += currentLight1Green;
  status += ",";
  status += currentLight1Blue;
  status += " LIGHT2 ";
  status += currentLight2Red;
  status += ",";
  status += currentLight2Green;
  status += ",";
  status += currentLight2Blue;
  return status;
}

String handleLightCommand(String payload) {
  const int firstSpace = payload.indexOf(' ');
  if (firstSpace < 0) {
    return "ERR Use LIGHT <1|2|ALL> <r> <g> <b>";
  }

  String target = payload.substring(0, firstSpace);
  target.trim();
  target.toUpperCase();

  String remainder = payload.substring(firstSpace + 1);
  remainder.trim();

  const int secondSpace = remainder.indexOf(' ');
  if (secondSpace < 0) {
    return "ERR Use LIGHT <1|2|ALL> <r> <g> <b>";
  }

  String redText = remainder.substring(0, secondSpace);
  remainder = remainder.substring(secondSpace + 1);
  remainder.trim();

  const int thirdSpace = remainder.indexOf(' ');
  if (thirdSpace < 0) {
    return "ERR Use LIGHT <1|2|ALL> <r> <g> <b>";
  }

  String greenText = remainder.substring(0, thirdSpace);
  String blueText = remainder.substring(thirdSpace + 1);
  redText.trim();
  greenText.trim();
  blueText.trim();

  long redValue = 0;
  long greenValue = 0;
  long blueValue = 0;
  if (!parseLong(redText, redValue) || !parseLong(greenText, greenValue) || !parseLong(blueText, blueValue)) {
    return "ERR Invalid light values";
  }

  const uint8_t red = clampColor(redValue);
  const uint8_t green = clampColor(greenValue);
  const uint8_t blue = clampColor(blueValue);

  if (target == "ALL") {
    setAllLights(red, green, blue);
    return buildStatus();
  }

  if (target == "1") {
    setLight(kPlayer1LightIndex, red, green, blue);
    return buildStatus();
  }

  if (target == "2") {
    setLight(kPlayer2LightIndex, red, green, blue);
    return buildStatus();
  }

  return "ERR Invalid light target";
}

String executeCommand(String command) {
  command.trim();
  if (command.length() == 0) {
    return "ERR Empty command";
  }

  if (command.indexOf('=') >= 0) {
    return postToScoreboard(command) ? "OK Forwarded legacy score command" : "ERR Scoreboard unreachable";
  }

  const int firstSpace = command.indexOf(' ');
  String action = firstSpace < 0 ? command : command.substring(0, firstSpace);
  String payload = firstSpace < 0 ? "" : command.substring(firstSpace + 1);
  action.toUpperCase();
  payload.trim();

  if (action == "SET" || action == "ADD" || action == "LCD" || action == "LCDCLEAR") {
    return postToScoreboard(command) ? "OK Forwarded" : "ERR Scoreboard unreachable";
  }

  if (action == "RESET") {
    setAllLights(0, 0, 0);
    return postToScoreboard("RESET") ? buildStatus() : "ERR Scoreboard unreachable";
  }

  if (action == "STATUS") {
    return buildStatus();
  }

  if (action == "LIGHT") {
    return handleLightCommand(payload);
  }

  if (action == "LIGHTOFF" || action == "LEDOFF") {
    setAllLights(0, 0, 0);
    return buildStatus();
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
    pinMode(buttons[i].pin, INPUT_PULLUP);
    const bool initial = digitalRead(buttons[i].pin);
    buttons[i].lastReading = initial;
    buttons[i].stableState = initial;
    buttons[i].lastDebounceAt = 0;
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
      if (button.action == ButtonAction::AddRight) {
        postToScoreboard(String("ADD 2 ") + button.delta);
      } else if (button.action == ButtonAction::TogglePlayer) {
        activePlayer = activePlayer == 1 ? 2 : 1;
        showActivePlayerLight();
      } else if (button.action == ButtonAction::ResetAll) {
        setAllLights(0, 0, 0);
        postToScoreboard("RESET");
      }

      Serial.println(button.message);
    }
  }

  button.lastReading = reading;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  setAllLights(0, 0, 0);
  initializeButtons();

  connectWifiBlocking();
  MDNS.begin(kHostname);
  setupServer();
  showActivePlayerLight();

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
