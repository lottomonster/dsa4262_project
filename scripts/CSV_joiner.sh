zcat $1 | head -n 1 |  awk -F',' '{print $1"_"$2","$0}' > header1.tmp
zcat $2 | head -n 1 |  awk -F',' '{print "ID_POS,"$0}' > header2.tmp

join -t ',' header1.tmp header2.tmp > combined.csv

zcat $1 | awk -F',' 'NR>1 {print $1"_"$2","$0}' | sort -t ',' -k 1 > 1.tmp
zcat $2 | awk -F',' 'NR>1 {print $1"_"$2","$0}' | sort -t ',' -k 1 > 2.tmp

join -t ',' 1.tmp 2.tmp >> combined.csv

rm header1.tmp header2.tmp 1.tmp 2.tmp

cut -d, combined.csv -f2-13,17 | gzip -c > combined.csv.gz
