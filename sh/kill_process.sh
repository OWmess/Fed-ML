#!/bin/bash

if [ -z "$1" ]
then
    echo "No port provided"
    exit 1
fi

PORT=$1
PID=$(lsof -t -i:$PORT)

if [ -z "$PID" ]
then
    echo "No process is using port $PORT"
else
    echo "Killing process $PID on port $PORT"
    kill -9 $PID
fi
