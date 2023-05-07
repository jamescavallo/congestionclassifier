current_date=$(date -u +%Y-%m-%d)
day=$(echo $current_date | cut -d'-' -f3)
month=$(echo $current_date | cut -d'-' -f2)
year=$(echo $current_date | cut -d'-' -f1)

touch /users/jcavallo/classify/data.csv
touch /users/jcavallo/classify/new_data.csv
rm -rf /users/jcavallo/classify/ndt-data/$year/$month/$day/

/users/jcavallo/tcp-info/tcp-info -output /users/jcavallo/classify/ndt-data -reps 100

zstd -df /users/jcavallo/classify/ndt-data/$year/$month/$day/*.jsonl.zst -o /users/jcavallo/classify/ndt-data/$year/$month/$day/output.jsonl

/users/jcavallo/tcp-info/cmd/csvtool/csvtool /users/jcavallo/classify/ndt-data/$year/$month/$day/output.jsonl > /users/jcavallo/classify/data.csv
