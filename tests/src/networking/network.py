import socketio
import time
import threading
import logging
from liveconfig import trigger

from src.core import config

logger = logging.getLogger(__name__)


class Network:
    def __init__(self):
        self.sio = socketio.Client()
        self.positions_requested = False
        self.finished_move = False
        self.gantry_moving = False
        self.finished_move_counter = 0
        self.finished_hit = False
        self.moving_to_origin = False
        self.obstruction_already_sent = False

        @self.sio.event
        def connect():
            try:
                self.sio.emit("join", "ballPositions")
                self.sio.emit("join", "obstructionDetected")
                self.sio.emit("join", "endOfTurn")
                self.sio.emit("join", "requestPositions")
                self.sio.emit("join", "correctedPositions")
                self.sio.emit("join", "finishedMove")
                self.sio.emit("join", "finishedHit")
                self.sio.emit("join", "move")
                logger.info("Connected to the server")
            except Exception as e:
                logger.error(f"Error during connection: {e}")

        @self.sio.event
        def disconnect():
            try:
                logger.info("Disconnected from the server")
                self.sio.disconnect()
            except Exception as e:
                logger.error(f"Error during disconnection: {e}")

        @self.sio.on("requestPositions")
        def handle_request_positions(data):
            self._handle_request_positions(data)

        @self.sio.on("finishedMove")
        def handle_finished_move(data):
            self._handle_finished_move(data)

        @self.sio.on("finishedHit")
        def handle_finished_hit(data):
            self._handle_finished_hit(data)

        @self.sio.on("move")
        def handle_move(data):
            self._handle_move(data)

    def reconnect(self):
        if self.sio.connected:
            logger.info("Already connected, no need to reconnect.")
            return
        threading.Thread(target=self._reconnect, daemon=True).start()

    def _reconnect(self):
        while not self.sio.connected:
            try:
                logger.info("Attempting to reconnect...")
                self.sio.connect(config.poolpal_url, wait=False)
                break
            except Exception as e:
                if "Client is not in a disconnected state" in str(e):
                    logger.info("Already connected, no need to reconnect.")
                    break
                logger.error(f"Reconnection failed: {e}")
                time.sleep(3)

    def connect(self):
        threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        try:
            self.sio.connect(config.poolpal_url, wait=False)
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.reconnect()

    def _handle_request_positions(self, data):
        self.positions_requested = True

    def _handle_finished_move(self, data):
        self.finished_move_counter += 1
        logger.info("Finished move")
        self.finished_move = True
        self.moving_to_origin = False
        self.gantry_moving = False

    def _handle_finished_hit(self, data):
        self.finished_hit = True
        logger.info("Hit finished, moving back to origin.")

    def _handle_move(self, data) -> None:
        self.gantry_moving = True
        if (int(data['x']) == 0 and int(data['y'] == 0)):
            self.moving_to_origin = True
        logger.info("Gantry moving.")

    def send_balls(self, balls):
        try:
            logger.info(f"Sending balls: {balls}")
            self.sio.emit("ballPositions", balls)
        except Exception as e:
            self._handle_error(e, "ballPositions")
            pass

    def send_corrected_white_ball(self, ball):
        try:
            logger.info(f"Sending corrected white ball: {ball}")
            self.sio.emit("correctedPositions", ball)
        except Exception as e:
            self._handle_error(e, "correctedPositions")
            pass

    def send_end_of_turn(self, end_of_turn):
        try:
            logger.info(f"Sending end of turn: {end_of_turn}")
            self.sio.emit("endOfTurn", end_of_turn)
        except Exception as e:
            self._handle_error(e, "endOfTurn")
            pass

    def send_obstruction(self, obstruction):
        try:
            logger.info(f"Sending obstruction: {obstruction}")
            self.sio.emit("obstructionDetected", obstruction)
        except Exception as e:
            self._handle_error(e, "obstructionDetected")
            pass

    def disconnect(self):
        self.sio.disconnect()

    def _handle_error(self, e, name):
        logger.error(f"Failed to send {name}: {e}")
