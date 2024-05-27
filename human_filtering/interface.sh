cd human_filtering
nohup flask --app back_end run --host=0.0.0.0 > flask.log 2>&1 &
cd front_end
nohup npm run dev > ../vue.log 2>&1 &
