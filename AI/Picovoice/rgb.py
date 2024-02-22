
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
for piece in pieces.keys():
	for led in pieces[piece]:
		GPIO.setup(led, GPIO.OUT, initial=GPIO.LOW)

for piece in pieces.keys():
	for couleur, indice in list(zip(['rouge','vert','bleu'],[RED,GREEN,BLUE])):
		print(f'Allume {piece} en {couleur}')
		switch_on(pieces[piece], indice)
		input('Eteindre')
		switch_off(pieces[piece])
input('fin')

GPIO.cleanup()


