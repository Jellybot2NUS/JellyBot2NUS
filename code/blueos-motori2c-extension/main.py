import requests
import time
from flask import Flask, jsonify, request

app = Flask(__name__)

MAVLINK2REST_URL = "http://localhost/mavlink2rest/v1/mavlink"


def set_servo_function(channel: int, function: int):
    """Set SERVO{channel}_FUNCTION parameter in ArduPilot."""
    payload = {
        "header": {"system_id": 255, "component_id": 0, "sequence": 0},
        "message": {
            "type": "PARAM_SET",
            "param_id": f"SERVO{channel}_FUNCTION",
            "param_value": float(function),
            "param_type": {"type": "MAV_PARAM_TYPE_INT32"},
            "target_system": 1,
            "target_component": 1
        }
    }
    response = requests.post(MAVLINK2REST_URL, json=payload)
    return response.status_code

def set_servo(channel: int, pwm: int):
    """
    Send a MAV_CMD_DO_SET_SERVO command via MAVLink2REST.
    channel: PWM output channel number (1-16)
    pwm: pulse width in microseconds
    """
    payload = {
        "header": {
            "system_id": 255,
            "component_id": 0,
            "sequence": 0
        },
        "message": {
            "type": "COMMAND_LONG",
            "param1": float(channel),
            "param2": float(pwm),
            "param3": 0.0,
            "param4": 0.0,
            "param5": 0.0,
            "param6": 0.0,
            "param7": 0.0,
            "command": {
                "type": "MAV_CMD_DO_SET_SERVO"
            },
            "target_system": 1,
            "target_component": 1,
            "confirmation": 0
        }
    }
    response = requests.post(MAVLINK2REST_URL, json=payload)
    return response.status_code, response.text

@app.route('/motor/on', methods=['POST'])
def motor_on():
    speed = int(request.args.get('speed', 100))
    pwm = int(1000 + (1000 * speed / 100))
    status, text = set_servo(channel=9, pwm=pwm)
    return jsonify({"status": "on", "speed": speed, "pwm": pwm, "response": text})

@app.route('/motor/off', methods=['POST'])
def motor_off():
    status, text = set_servo(channel=9, pwm=800)
    return jsonify({"status": "off", "response": text})

if __name__ == '__main__':
    print("Setting SERVO9_FUNCTION to 0 (disabled)...")
    time.sleep(5)
    status = set_servo_function(9, 0)
    print(f"SERVO9_FUNCTION set — response status: {status}")
    app.run(host='0.0.0.0', port=8082)