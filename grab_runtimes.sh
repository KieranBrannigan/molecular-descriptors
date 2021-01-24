for f in $(ls $1)
do
    echo "$(basename $f .log),$(grep 'Job cpu time' "$1/$f")"
done