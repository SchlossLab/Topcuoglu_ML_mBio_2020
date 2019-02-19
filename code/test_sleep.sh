while :; do
    count="$(ls -1 data/temp/ | wc -l)"
    echo "$count";
    if [ "$count" == 2101 ]; then
        break
    fi;
    sleep 1m;
done
