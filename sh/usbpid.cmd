@echo off
usbipd list
set /p busid="Please enter the busid: "

usbipd bind --busid %busid%
usbipd attach --wsl --busid %busid%