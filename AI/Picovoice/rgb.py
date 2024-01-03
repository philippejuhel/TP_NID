
## IMPORTANT : this program must be run as super user like this : 
## sudo python rgb.py

import RPi.GPIO as GPIO

RED = 0
GREEN = 1
BLUE = 2

LED_CUISINE = [16,20,21] # GPIO for RGB
LED_SALON = [17,27,22]
LED_CHAMBRE = [5,6,13] 
pieces = {'cuisine':LED_CUISINE, 'salon':LED_SALON, 'chambre':LED_CHAMBRE}


def switch_off(room):
	for led in room:
		GPIO.output(led, GPIO.LOW)

def switch_on(room, color):
	switch_off(room)
	GPIO.output(room[color], GPIO.HIGH)
	

# Main program
GPIO.setmode(GPIO.BCM)

# Switch off all LEDs
for piece in pieces:
	for led in piece:
		GPIO.setup(led, GPIO.OUT, initial=GPIO.LOW)

print('Allume cuisine en vert')
switch_on(LED_CUISINE, GREEN)
input('Eteindre')
switch_off(LED_CUISINE)
print('Allume salon en bleu')
switch_on(LED_SALON, BLUE)
input('Eteindre')
switch_off(LED_SALON)
print('Allume chambre en rouge')
switch_on(LED_CHAMBRE, RED)
input('fin')

GPIO.cleanup()


