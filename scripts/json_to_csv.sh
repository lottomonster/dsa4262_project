if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_json_file> <output_csv_file>"
    exit 1
fi

input="$1"
output="$2"

mkdir -p ./data/csv/

if [[ $input == *.gz ]]; then
    reader="zcat"
else
    reader="cat"
fi

$reader "$input" |
    awk -F "(:|])" '{gsub(/[{}"[]/, "", $0)
        for (i=4;i<=NF;i++) {if (length($i) > 0){
		n=split($i, a, ",");
		printf "%s,%s,%s", $1, $2, $3;
		{for (j=1; j<=n; j++){
			if (length(a[j]) > 1){printf ",%s", a[j]};};
		printf "\n"}}}}' >  $2
sed -i '1i\transcript_id,transcript_position,7mer,PreTime,PreSD,PreMean,InTime,InSD,InMean,PostTime,PostSD,PostMean' $2
