zcat $1 |
    awk -F "(:|])" '{gsub(/[{}"[]/, "", $0)
        for (i=4;i<=NF;i++) {if (length($i) > 0){
		n=split($i, a, ",");
		printf "%s,%s,%s", $1, $2, $3;
		{for (j=1; j<=n; j++){
			if (length(a[j]) > 1){printf ",%s", a[j]};};
		printf "\n"}}}}' >  $2

sed -i '1i\ID,POS,SEQ,PreTime,PreSD,PreMean,InTime,InSD,InMean,PostTime,PostSD,PostMean' $2
gzip $2
