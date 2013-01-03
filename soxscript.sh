#!/bin/bash
./sox -r 16000 -b 16 -c 1 -e signed-integer ../sms2012/left.raw ../sms2012/left.wav
./sox -r 16000 -b 16 -c 1 -e signed-integer ../sms2012/right.raw ../sms2012/right.wav
./sox left.wav left.dat
./sox right.wav right.dat
./play -e signed-integer -r 16000 output.dat
