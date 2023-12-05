
import RPi.GPIO as GPIO
import time

# Pin Definitions
class_pin = [12,16,18]  # BCM pin 18, BOARD pin 12

# Pin Setup:
GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme from Raspberry Pi

# set pin as an output pin with optional initial state of LOW
for i in range(len(class_pin)):
    GPIO.setup(class_pin[i], GPIO.OUT, initial=GPIO.LOW)

# Switch on then off each LEDs
for i in class_pin:
    GPIO.output(i, True)  # switch on the LED
    time.sleep(1)
    GPIO.output(i, False) # Switch off the LED
    time.sleep(1)




