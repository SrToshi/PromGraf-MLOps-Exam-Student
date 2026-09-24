#!/bin/bash
# Simple traffic generator for the /predict endpoint.
# Sends N requests with randomized, realistic feature values.

API_URL="http://localhost:8080/predict"
NUM_REQUESTS=${1:-30}  # default to 30 requests if no argument given

echo "Sending $NUM_REQUESTS randomized requests to $API_URL..."

for i in $(seq 1 "$NUM_REQUESTS"); do
  temp=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); printf "%.2f", rand()}')
  atemp=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); printf "%.2f", rand()}')
  hum=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); printf "%.2f", rand()}')
  windspeed=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); printf "%.2f", rand()}')
  mnth=$(( (RANDOM % 12) + 1 ))
  hr=$(( RANDOM % 24 ))
  weekday=$(( RANDOM % 7 ))
  season=$(( (RANDOM % 4) + 1 ))
  holiday=$(( RANDOM % 2 ))
  workingday=$(( RANDOM % 2 ))
  weathersit=$(( (RANDOM % 4) + 1 ))

  curl -s -o /dev/null -w "Request %{http_code}\n" -X POST "$API_URL" \
    -H 'Content-Type: application/json' \
    -d "{\"temp\": $temp, \"atemp\": $atemp, \"hum\": $hum, \"windspeed\": $windspeed, \"mnth\": $mnth, \"hr\": $hr, \"weekday\": $weekday, \"season\": $season, \"holiday\": $holiday, \"workingday\": $workingday, \"weathersit\": $weathersit, \"dteday\": \"2011-01-15\"}"

  sleep 0.2
done

echo "Done."